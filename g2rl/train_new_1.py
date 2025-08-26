#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union
from collections import defaultdict
import os, sys, time, inspect, random

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from pogema import AStarAgent

# ================== 路径与项目模块 ==================
project_root = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main - train"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from g2rl.environment import G2RLEnv
from g2rl.agent import DDQNAgent
from g2rl.network import CRNNModel
from g2rl import moving_cost, detour_percentage  # 若未用到也保留，便于扩展

# ================== 通用工具 ==================
def get_timestamp() -> str:
    return datetime.now().strftime('%H-%M-%d-%m-%Y')

def _np_grid(g):
    arr = np.array(g, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("grid must be 2D")
    return (arr > 0).astype(np.uint8)

# def build_env_from_raw(raw_cfg: dict) -> G2RLEnv:
#     """仅将 G2RLEnv.__init__ 支持的键传入；grid/starts/goals 等挂属性。"""
#     sig = inspect.signature(G2RLEnv.__init__)
#     allowed = {
#         p.name for p in sig.parameters.values()
#         if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
#     }
#     allowed.discard("self")
#     rename = {}  # 如果 YAML 键名与 __init__ 不一致，可在这里做映射

#     ctor_cfg = {}
#     for k, v in raw_cfg.items():
#         kk = rename.get(k, k)
#         if kk in allowed:
#             ctor_cfg[kk] = v

#     env = G2RLEnv(**ctor_cfg)

#     # 将 grid/starts/goals 作为属性挂载（不传入 __init__）
#     if "grid" in raw_cfg:
#         try:
#             env.grid = _np_grid(raw_cfg["grid"])
#         except Exception:
#             env.grid = None
#     if "starts" in raw_cfg:
#         env.starts = raw_cfg["starts"]
#     if "goals" in raw_cfg:
#         env.goals = raw_cfg["goals"]
#     return env
#


# test动画用
def build_env_from_raw(raw_cfg: dict) -> G2RLEnv:
    sig = inspect.signature(G2RLEnv.__init__)
    allowed = {
        p.name for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    allowed.discard("self")

    # 这里默认关闭常见“目标重生/重分配/重排”开关（如果构造器支持）
    defaults_to_disable = {
        "respawn_goals": False,
        "reassign_goals": False,
        "shuffle_agents": False,
        "reset_goals_each_step": False,
    }

    ctor_cfg = {}
    for k, v in raw_cfg.items():
        if k in allowed:
            ctor_cfg[k] = v
    for k, v in defaults_to_disable.items():
        if (k in allowed) and (k not in ctor_cfg):
            ctor_cfg[k] = v

    env = G2RLEnv(**ctor_cfg)

    # grid/starts/goals 仍按原逻辑挂属性
    if "grid" in raw_cfg:
        try:
            env.grid = _np_grid(raw_cfg["grid"])
        except Exception:
            env.grid = None
    if "starts" in raw_cfg:
        env.starts = raw_cfg["starts"]
    if "goals" in raw_cfg:
        env.goals = raw_cfg["goals"]
    return env



# ================== 特征抽取（用于“难度评分”与分析） ==================
def extract_features_from_spec(name: str, spec: dict) -> Dict[str, float]:
    """
    从 YAML 条目与 grid 中抽取特征。
    保留 Size / Agents / Density / Density_actual + FRA/FDA/BN/LDD/MC/DLR（若存在）。
    """
    feats = {}
    feats["map_name"]   = name
    feats["Size"]       = float(spec.get("size", spec.get("Size", np.nan)))
    feats["Agents"]     = float(spec.get("num_agents", spec.get("Agents", np.nan)))
    feats["Density"]    = float(spec.get("density", spec.get("Density", np.nan)))

    # density_actual：按 grid 真实障碍占比（有 grid 时计算）
    try:
        if "grid" in spec and spec["grid"] is not None:
            g = _np_grid(spec["grid"])
            feats["Density_actual"] = float(g.mean())
        else:
            feats["Density_actual"] = np.nan
    except Exception:
        feats["Density_actual"] = np.nan

    # 其他特征：如果 YAML 已算好就带上
    for k in ["FRA", "FDA", "BN", "LDD", "MC", "DLR"]:
        if k in spec:
            try:
                feats[k] = float(spec[k])
            except Exception:
                feats[k] = np.nan
        else:
            feats[k] = np.nan
    return feats

# ================== 成功判定辅助 ==================
def _bool_from(x, idx: int) -> bool:
    """把 terminated/truncated 规整为目标代理的布尔。"""
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0: 
            return False
        if 0 <= idx < len(x):
            return bool(x[idx])
        return bool(any(x))
    if isinstance(x, dict):
        cand_keys = [idx, "global", "any", "done", "terminated"]
        for key in cand_keys:
            if key in x:
                try:
                    return bool(x[key])
                except Exception:
                    pass
        try:
            return bool(next(iter(x.values())))
        except Exception:
            return False
    try:
        return bool(x)
    except Exception:
        return False

def _success_from_info(info, idx: int) -> Optional[bool]:
    """从 info 中抽取成功信号，抽不到返回 None。"""
    if not isinstance(info, dict):
        return None
    cand = info.get(idx, info)  # 有些 env 为每个 agent 提供子 dict
    if isinstance(cand, dict):
        for k in ("success", "is_success", "solved", "reached_goal", "done"):
            if k in cand:
                try:
                    return bool(cand[k])
                except Exception:
                    pass
    return None

# ================== 单集评估（带超时与步数上限） ==================
# test动画
# === REPLACE your run_one_episode with this ===
def run_one_episode(
    env: G2RLEnv,
    agent: DDQNAgent,
    episode_idx: int,
    max_episode_seconds: int = 30,
    max_steps_cap: Optional[int] = None,
) -> Dict[str, float]:
    try:
        obs, info = env.reset()
    except Exception:
        obs = env.reset()
        info = {}

    # ① 固定同一个 agent
    target_idx = np.random.randint(env.num_agents)
    target_id  = int(obs[target_idx].get('agent_id', target_idx))

    # ② 只在 reset 后取一次“绝对终点”，作为 fixed_goal
    try:
        fixed_goal = tuple(map(int, env.goals[target_idx]))
    except Exception:
        fixed_goal = None

    # ③ 其他队友用 A*
    teammates = [agent if i == target_idx else AStarAgent() for i in range(env.num_agents)]

    # 参考最短路径长度（可选）
    state = obs[target_idx]
    try:
        opt_path = [state['global_xy']] + env.global_guidance[target_idx]
        opt_len = max(1, len(opt_path) - 1)
    except Exception:
        opt_len = 1

    # 步数预算
    timesteps_per_episode = 50 + 10 * episode_idx
    max_steps_budget = min(timesteps_per_episode, max_steps_cap) if isinstance(max_steps_cap, int) and max_steps_cap > 0 else timesteps_per_episode

    t0 = time.time()
    success, steps, timeout_flag = 0, 0, 0

    for _ in range(max_steps_budget):
        steps += 1
        if time.time() - t0 > max_episode_seconds:
            timeout_flag = 1
            break

        # 若 obs 顺序会变，用 agent_id 找回同一名代理
        if 'agent_id' in obs[0]:
            target_idx = next(i for i, o in enumerate(obs) if int(o.get('agent_id', -1)) == target_id)

        # A) 到达“前”判定（有些 env 会在 step() 时立刻换目标）
        try:
            pos_before = tuple(map(int, obs[target_idx]['global_xy']))
        except Exception:
            pos_before = None
        if (fixed_goal is not None) and (pos_before is not None) and (pos_before == fixed_goal):
            success = 1
            break

        # 行动
        actions = [ag.act(o) for ag, o in zip(teammates, obs)]
        out = env.step(actions)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            obs, reward, done, info = out
            terminated, truncated = done, False

        # B) 到达“后”立即判定，用 fixed_goal（不要用会变的 global_target_xy）
        try:
            pos_after = tuple(map(int, obs[target_idx]['global_xy']))
        except Exception:
            pos_after = None
        if (fixed_goal is not None) and (pos_after is not None) and (pos_after == fixed_goal):
            success = 1
            break

        # C) （仅用于你的显示/日志）尝试把 env.goals 锁回去，避免你后面打印看到“变了”
        try:
            env.goals[target_idx] = fixed_goal
        except Exception:
            pass

        # 经验回放/学习
        r_target = reward[target_idx] if isinstance(reward, (list, tuple, np.ndarray)) else reward
        agent.store(
            state,
            actions[target_idx],
            r_target,
            obs[target_idx],
            bool(terminated[target_idx] if isinstance(terminated, (list, tuple, np.ndarray)) else terminated),
        )
        state = obs[target_idx]

    return {
        "success": int(success),
        "steps": int(steps),
        "opt_len": float(opt_len),
        "timeout": int(timeout_flag or (steps >= max_steps_budget)),
    }
# === END REPLACE ===

# def run_one_episode(
#     env: G2RLEnv,
#     agent: DDQNAgent,
#     episode_idx: int,
#     max_episode_seconds: int = 30,
#     max_steps_cap: Optional[int] = None,
# ) -> Dict[str, float]:
#     try:
#         obs, info = env.reset()
#     except Exception:
#         obs = env.reset()
#         info = {}

#     # ① 绑定固定的目标代理
#     target_idx = np.random.randint(env.num_agents)
#     target_id  = int(obs[target_idx].get('agent_id', target_idx))  # 若无 agent_id，就用索引兜底

#     # ② 固定一次性的“绝对目标坐标”（绝不再改）
#     try:
#         fixed_goal = tuple(map(int, env.goals[target_idx]))
#     except Exception:
#         fixed_goal = None

#     # ③ 其余队友用 A*
#     teammates = [agent if i == target_idx else AStarAgent() for i in range(env.num_agents)]

#     # 参考最优路径长度（可选）
#     state = obs[target_idx]
#     try:
#         opt_path = [state['global_xy']] + env.global_guidance[target_idx]
#         opt_len = max(1, len(opt_path) - 1)
#     except Exception:
#         opt_len = 1

#     # 步数预算
#     timesteps_per_episode = 50 + 10 * episode_idx
#     max_steps_budget = min(timesteps_per_episode, max_steps_cap) if isinstance(max_steps_cap, int) and max_steps_cap > 0 else timesteps_per_episode

#     t0 = time.time()
#     success, steps, timeout_flag = 0, 0, 0

#     for _ in range(max_steps_budget):
#         steps += 1
#         if time.time() - t0 > max_episode_seconds:
#             timeout_flag = 1
#             break

#         # 若 obs 顺序可能变化：用 agent_id 找回同一名代理的索引
#         if 'agent_id' in obs[0]:
#             target_idx = next(i for i, o in enumerate(obs) if int(o.get('agent_id', -1)) == target_id)

#         actions = [ag.act(o) for ag, o in zip(teammates, obs)]
#         out = env.step(actions)
#         if len(out) == 5:
#             obs, reward, terminated, truncated, info = out
#         else:
#             obs, reward, done, info = out
#             terminated, truncated = done, False

#         # 只用绝对坐标与“固定目标”判定成功
#         try:
#             pos = tuple(map(int, obs[target_idx]['global_xy']))
#         except Exception:
#             pos = None

#         if (fixed_goal is not None) and (pos is not None) and (pos == fixed_goal):
#             success = 1
#             break

#         # 兜底：info 若明确给成功信号
#         if isinstance(info, dict):
#             sub = info.get(target_idx, info)
#             if isinstance(sub, dict) and any(k in sub and bool(sub[k]) for k in ('success', 'is_success', 'reached_goal', 'done')):
#                 success = 1
#                 break

#         # 兼容 done 信号：再对比一次固定目标
#         if (isinstance(terminated, (list, tuple, np.ndarray)) and 0 <= target_idx < len(terminated) and terminated[target_idx]) \
#            or (isinstance(truncated, (list, tuple, np.ndarray)) and 0 <= target_idx < len(truncated) and truncated[target_idx]):
#             if (fixed_goal is not None) and (pos is not None) and (pos == fixed_goal):
#                 success = 1
#             break

#         # 经验回放（如需在线学习）
#         r_target = reward[target_idx] if isinstance(reward, (list, tuple, np.ndarray)) else reward
#         agent.store(state, actions[target_idx], r_target, obs[target_idx],
#                     bool(terminated[target_idx] if isinstance(terminated, (list, tuple, np.ndarray)) else terminated))
#         state = obs[target_idx]

#     return {"success": int(success), "steps": int(steps), "opt_len": float(opt_len),
#             "timeout": int(timeout_flag or (steps >= max_steps_budget))}

# ================== “难度评分” & 课程构建（不使用 complexity） ==================
def compute_difficulty_from_features(df_feats: pd.DataFrame) -> pd.DataFrame:
    """
    计算一个可控的 difficulty_score：
      Agents↑、Density/Density_actual↑、Size↑ => 难度↑
      FRA/FDA↑（更易通行） => 难度↓
    做 0~1 归一化后线性加权，便于调整。
    """
    d = df_feats.copy()

    dens_act = d["Density_actual"] if "Density_actual" in d else d["Density"]

    cols = {
        "Size": d.get("Size", pd.Series(np.nan, index=d.index)),
        "Agents": d.get("Agents", pd.Series(np.nan, index=d.index)),
        "DensityA": dens_act,
        "Density": d.get("Density", pd.Series(np.nan, index=d.index)),
        "FRA": d.get("FRA", pd.Series(np.nan, index=d.index)),
        "FDA": d.get("FDA", pd.Series(np.nan, index=d.index)),
    }
    X = pd.DataFrame(cols)

    def _minmax(s: pd.Series):
        s = s.astype(float)
        if s.notna().sum() <= 1:
            return s
        lo, hi = np.nanmin(s.values), np.nanmax(s.values)
        if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
            return s
        return (s - lo) / (hi - lo)

    Xn = X.apply(_minmax)

    # 权重（可按需微调）
    w_agents   = 0.50
    w_density  = 0.30
    w_size     = 0.20
    w_fra_good = 0.10
    w_fda_good = 0.10

    dens_for_score = Xn["DensityA"].fillna(Xn["Density"])

    diff = (
        w_agents  * Xn["Agents"].fillna(0.5) +
        w_density * dens_for_score.fillna(0.5) +
        w_size    * Xn["Size"].fillna(0.5) -
        w_fra_good * Xn["FRA"].fillna(0.0) -
        w_fda_good * Xn["FDA"].fillna(0.0)
    )

    d["difficulty_score"] = diff.values
    med = np.nanmedian(d["difficulty_score"])
    d["difficulty_score"] = d["difficulty_score"].fillna(med if np.isfinite(med) else 0.5)
    return d

def build_curriculum_buckets(maps_cfg: Dict[str, dict], n_stages: int = 5, min_per_stage: int = 1):
    """
    1) 提取每图特征 → 2) 计算 difficulty_score → 3) 排序 → 4) 分位数切割成 n 个 stage。
    返回：List[List[(map_name, spec, feature_row_dict)]] 与 df_scored（含 difficulty_score）。
    """
    rows = []
    for name, spec in maps_cfg.items():
        feats = extract_features_from_spec(name, spec)
        rows.append(feats)
    df_feats = pd.DataFrame(rows)

    df_scored = compute_difficulty_from_features(df_feats).sort_values("difficulty_score").reset_index(drop=True)

    qs = np.linspace(0, 1, n_stages + 1)
    edges = np.quantile(df_scored["difficulty_score"].values, qs)
    buckets = []
    for i in range(n_stages):
        lo, hi = float(edges[i]), float(edges[i+1]) + 1e-12
        sub = df_scored[(df_scored["difficulty_score"] >= lo) & (df_scored["difficulty_score"] < hi)]
        if len(sub) < min_per_stage and len(df_scored) >= min_per_stage:
            need = min_per_stage - len(sub)
            center = (lo + hi) / 2.0
            extra = df_scored.iloc[(df_scored["difficulty_score"] - center).abs().argsort()[:need]]
            sub = pd.concat([sub, extra]).drop_duplicates(subset=["map_name"])

        bucket = []
        for _, r in sub.iterrows():
            name = str(r["map_name"])
            bucket.append((name, maps_cfg[name], r.to_dict()))

        # 去重
        seen, uniq = set(), []
        for item in bucket:
            if item[0] not in seen:
                uniq.append(item)
                seen.add(item[0])
        buckets.append(uniq)

    return buckets, df_scored

# ================== 可视化：Features vs Success ==================
def make_feature_success_plots(out_dir: Union[str, Path], df: pd.DataFrame):
    """
    生成：
    - success_rate by map 的柱状图
    - 每个特征 vs success_rate 的散点图
    - 特征+成功率的相关性热图
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 每图成功率柱状图
    try:
        df_sorted = df.sort_values("success_rate")
        plt.figure(figsize=(max(8, 0.35*len(df_sorted)), 5))
        plt.bar(df_sorted["map_name"].astype(str), df_sorted["success_rate"].values)
        plt.xticks(rotation=60, ha="right")
        plt.ylabel("Success Rate")
        plt.title("Success Rate by Map")
        plt.tight_layout()
        plt.savefig(out_dir / "sr_by_map.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"[Plot] sr_by_map.png 失败: {e}")

    # 2) 每个数值特征 vs success_rate 散点图
    numeric_cols = []
    for c in df.columns:
        if c in ["map_name", "success_rate", "stage", "difficulty_score"]:
            continue
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_cols.append(c)
        except Exception:
            pass

    for c in numeric_cols:
        try:
            d = df.dropna(subset=[c, "success_rate"])
            if len(d) == 0:
                continue
            plt.figure(figsize=(6, 4))
            plt.scatter(d[c].values, d["success_rate"].values, s=18)
            plt.xlabel(c)
            plt.ylabel("Success Rate")
            plt.title(f"Success Rate vs {c}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f"success_vs_{c}.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"[Plot] success_vs_{c}.png 失败: {e}")

    # 3) 相关性热图
    try:
        # 用每图聚合后的数据（避免每阶段重复）
        corr_cols = ["success_rate"] + [c for c in numeric_cols if c in df.columns]
        dnum = df[corr_cols].copy()
        dnum = dnum.apply(pd.to_numeric, errors="coerce")
        cm = dnum.corr().values
        labels = dnum.columns.tolist()

        plt.figure(figsize=(1.2*len(labels), 1.0*len(labels)))
        im = plt.imshow(cm, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.title("Correlation (numeric features & success_rate)")
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "correlation_heatmap.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"[Plot] correlation_heatmap.png 失败: {e}")

# ================== 课程训练（由易到难，覆盖全部地图） ==================
def train_with_curriculum(
    model: torch.nn.Module,
    buckets: List[List[tuple]],
    episodes_per_map_stage: int = 100,  # 每阶段、每图的集数
    batch_size: int = 64,
    replay_buffer_size: int = 100_000,
    decay_range: int = 50_000,
    lr: float = 5e-4,
    device: str = "cuda",
    log_dir: str = "logs",
    run_dir: Optional[str] = None,
    max_episode_seconds: int = 60,
    max_steps_cap: Optional[int] = 3000,
) -> pd.DataFrame:
    """
    Stage 0(简单) → ... → Stage N-1(困难)；
    每阶段对该桶内所有地图各跑 episodes_per_map_stage 集。
    """
    run_dir = Path(run_dir or (Path(log_dir) / get_timestamp()))
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    # 用第一张图的动作空间初始化 agent
    first_name, first_spec, _ = buckets[0][0]
    first_env = build_env_from_raw(first_spec)
    agent = DDQNAgent(
        model,
        first_env.get_action_space(),
        lr=lr,
        decay_range=decay_range,
        device=device,
        replay_buffer_size=replay_buffer_size,
    )

    global_rows = []

    for stage_id, bucket in enumerate(buckets):
        print(f"\n====== Stage {stage_id}（{len(bucket)} maps）======")
        for map_name, spec, feats in bucket:
            env = build_env_from_raw(spec)
            pbar = tqdm(range(episodes_per_map_stage), desc=f"Stage{stage_id}[{map_name}]", dynamic_ncols=True)
            succ = 0; timeouts = 0
            for ep in pbar:
                out = run_one_episode(
                    env, agent, ep,
                    max_episode_seconds=max_episode_seconds,
                    max_steps_cap=max_steps_cap
                )
                succ += out["success"]; timeouts += out["timeout"]

                # 在线学习
                if len(agent.replay_buffer) >= batch_size:
                    _ = agent.retrain(batch_size)

                # TB
                writer.add_scalar(f"stage_{stage_id}/{map_name}_success", out["success"], ep)
                writer.add_scalar(f"stage_{stage_id}/{map_name}_steps", out["steps"], ep)
                writer.add_scalar(f"stage_{stage_id}/{map_name}_timeout", out["timeout"], ep)

                pbar.set_postfix(sr=f"{succ/(ep+1):.2f}")

            # 记录该图在本阶段表现
            row = dict(feats)
            row.update({
                "stage": stage_id,
                "map_name": map_name,
                "success_rate": succ / max(1, episodes_per_map_stage),
                "timeout_rate": timeouts / max(1, episodes_per_map_stage),
            })
            global_rows.append(row)

    df_all = pd.DataFrame(global_rows)
    out_csv = run_dir / "curriculum_features_vs_success_1.csv"
    df_all.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"📝 已保存：{out_csv}")

    # 聚合到“每图一行”用于可视化（成功率取均值，其它特征取第一条）
    keep_cols = ["Size","Agents","Density","Density_actual","FRA","FDA","BN","LDD","MC","DLR","difficulty_score"]
    agg_dict = {"success_rate":"mean", "timeout_rate":"mean"}
    for c in keep_cols:
        if c in df_all.columns:
            agg_dict[c] = "first"
    df_map = df_all.groupby("map_name", as_index=False).agg(agg_dict)

    try:
        make_feature_success_plots(run_dir, df_map)
        print(f"📊 可视化输出到：{run_dir}")
    except Exception as e:
        print(f"⚠️ 可视化失败：{e}")

    # 打印全局均值（按每图均值再取平均）
    global_sr = df_map["success_rate"].mean() if len(df_map) else 0.0
    print(f"🌍 Global Success Rate (mean of per-map SR): {global_sr:.3f}")

    writer.close()
    return df_all

# ================== 入口 ==================
if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 读取 YAML
    MAP_SETTINGS_PATH = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main - train/g2rl/map_settings_generated_new.yaml"
    with open(MAP_SETTINGS_PATH, "r", encoding="utf-8") as f:
        base_map_settings = yaml.safe_load(f)

    # 顶层 list → dict
    if isinstance(base_map_settings, list):
        base_map_settings = {
            (m.get("name") or f"map_{i}"): m
            for i, m in enumerate(base_map_settings)
        }

    # 构建课程（由易到难，不用 complexity）
    buckets, df_scored = build_curriculum_buckets(
        base_map_settings,
        n_stages=5,       # 难度阶段数
        min_per_stage=1,  # 每阶段至少几张图
    )
    print("📚 Curriculum buckets（由易到难）:")
    for i, bk in enumerate(buckets):
        names = [n for (n, _, __) in bk]
        print(f"  Stage {i}: {names}")

    # 设备 & 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNNModel().to(device)

    # 由易到难进行课程训练（覆盖全部地图）
    df_all = train_with_curriculum(
        model=model,
        buckets=buckets,
        episodes_per_map_stage=256,                     # 每阶段、每张图的 episode 数
        batch_size=64,
        replay_buffer_size=100_000,
        decay_range=50_000,
        lr=5e-4,
        device=device,
        log_dir='logs',
        run_dir=r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main - train/curriculum_run",
        max_episode_seconds=60,                         # 墙钟超时
        max_steps_cap=3000,                             # 步数上限（None 关闭）
    )

    # 保存模型
    torch.save(model.state_dict(), 'models/best_model_1.pt')
    print('✅ 模型已保存到 models/best_model_1.pt')


