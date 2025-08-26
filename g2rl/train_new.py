#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union
from collections import defaultdict
import os, sys, time, inspect, random, csv

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from pogema import AStarAgent

# ============ 项目根路径 ============
project_root = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main - train"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============ 项目模块 ============
from g2rl.environment import G2RLEnv
from g2rl.agent import DDQNAgent
from g2rl.network import CRNNModel
from g2rl import moving_cost, detour_percentage

# ============ 基础工具 ============
def get_timestamp() -> str:
    return datetime.now().strftime('%H-%M-%d-%m-%Y')

def _np_grid(g):
    arr = np.array(g, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("grid must be 2D")
    return (arr > 0).astype(np.uint8)

def build_env_from_raw(raw_cfg: dict) -> G2RLEnv:
    """只把 G2RLEnv.__init__ 支持的键传入，grid/starts/goals 等挂属性。"""
    sig = inspect.signature(G2RLEnv.__init__)
    allowed = {
        p.name for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    allowed.discard("self")
    rename = {}  # 如需要做键名映射可在此补充（例如 {'size':'map_size'}）

    ctor_cfg = {}
    for k, v in raw_cfg.items():
        kk = rename.get(k, k)
        if kk in allowed:
            ctor_cfg[kk] = v

    env = G2RLEnv(**ctor_cfg)

    # 将 grid/starts/goals 作为属性挂载（不进入 __init__）
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

def extract_features_from_spec(name: str, spec: dict) -> Dict[str, float]:
    """
    从 YAML spec 和 grid 中抽取用于分析的 features。
    保留 Size / Agents / Density / Density_actual + FRA/FDA/BN/LDD/MC/DLR（若 YAML 已有）。
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

    # 其他可选特征：FRA/FDA/BN/LDD/MC/DLR（如果 YAML 已算好就带上）
    for k in ["FRA", "FDA", "BN", "LDD", "MC", "DLR"]:
        if k in spec:
            try:
                feats[k] = float(spec[k])
            except Exception:
                feats[k] = np.nan
        else:
            feats[k] = np.nan
    return feats

# ============ 成功判定辅助 ============
def _bool_from(x, idx: int) -> bool:
    """把 terminated/truncated 这类多代理返回规整为单个 target 的 bool。"""
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0: 
            return False
        if 0 <= idx < len(x):
            return bool(x[idx])
        return bool(any(x))
    if isinstance(x, dict):
        # 常见字段尝试
        for key in (idx, "global", "any", "done", "terminated"):
            if key in x:
                try: return bool(x[key])
                except Exception: pass
        try:
            return bool(next(iter(x.values())))
        except Exception:
            return False
    try:
        return bool(x)
    except Exception:
        return False

def _success_from_info(info, idx: int) -> Optional[bool]:
    """从 info 中尽可能抽取成功信号；抽不到返回 None。"""
    if not isinstance(info, dict):
        return None
    cand = info.get(idx, info)  # 有些 env 会按 agent 索引给子 dict
    if isinstance(cand, dict):
        for k in ("success", "is_success", "solved", "reached_goal", "done"):
            if k in cand:
                try:
                    return bool(cand[k])
                except Exception:
                    pass
    return None

# ============ 评估：单集 ============
def run_one_episode(
    env: G2RLEnv, 
    agent: DDQNAgent, 
    episode_idx: int,
    max_episode_seconds: int = 30,
    max_steps_cap: Optional[int] = None
) -> Dict[str, float]:
    """
    单集 rollout：选一个 target agent（其余用 A*）。
    成功判定优先顺序：
      1) 目标代理位置 == 自身 goal
      2) env 的 info/terminated/truncated 给出的成功信号
    同时启用两种“超时保护”：墙钟时间 + 步数上限（任何一个触发即失败）
    """
    try:
        obs, info = env.reset()
    except Exception:
        obs = env.reset()
        info = {}

    # 选一个目标代理：默认随机（也可以改 shortest guidance）
    target_idx = np.random.randint(env.num_agents)
    teammates = [agent if i == target_idx else AStarAgent() for i in range(env.num_agents)]

    # 获取目标/参考最短长度（如果 env 提供 global_guidance）
    try:
        goal = tuple(env.goals[target_idx])
    except Exception:
        goal = None

    state = obs[target_idx]
    try:
        opt_path = [state['global_xy']] + env.global_guidance[target_idx]
        opt_len = max(1, len(opt_path) - 1)
    except Exception:
        opt_len = 1

    # -------- 超时上限 --------
    # 基于 episode_idx 的回合上限（你原逻辑）
    timesteps_per_episode = 50 + 10 * episode_idx
    # 允许传入更强的步数上限（优先取更小的那个，以尽快止损）
    if isinstance(max_steps_cap, int) and max_steps_cap > 0:
        max_steps_budget = min(timesteps_per_episode, max_steps_cap)
    else:
        max_steps_budget = timesteps_per_episode

    t0 = time.time()
    success = 0
    steps = 0
    timeout_flag = 0

    for t in range(max_steps_budget):
        steps += 1
        if time.time() - t0 > max_episode_seconds:
            timeout_flag = 1
            break

        actions = [ag.act(o) for ag, o in zip(teammates, obs)]
        out = env.step(actions)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            # 兼容旧式 (obs, reward, done, info)
            obs, reward, done, info = out
            terminated, truncated = done, False

        # --- 成功判定 #1：位置到达目标 ---
        try:
            pos = tuple(obs[target_idx]['global_xy'])
        except Exception:
            pos = None
        if (goal is not None) and (pos is not None) and (pos == goal):
            success = 1
            break

        # --- 成功判定 #2：从 info / terminated / truncated 中读信号 ---
        s_from_info = _success_from_info(info, target_idx)
        if s_from_info is True:
            success = 1
            break

        # 如果 env 提早结束也要退出（按需可放宽）
        if _bool_from(terminated, target_idx) or _bool_from(truncated, target_idx):
            # 若没读到成功信号，再兜底用位置
            if success == 0 and (goal is not None) and (pos is not None) and (pos == goal):
                success = 1
            break

        # 经验回放 + 在线学习
        agent.store(
            state,
            actions[target_idx],
            reward[target_idx] if isinstance(reward, (list, tuple, np.ndarray)) else reward,
            obs[target_idx],
            bool(_bool_from(terminated, target_idx)),
        )
        state = obs[target_idx]

    return {
        "success": int(success),
        "steps": int(steps),
        "opt_len": float(opt_len),
        "timeout": int(timeout_flag or (steps >= max_steps_budget)),
    }

# ============ 主训练/评估：按图统计成功率 ============
def train(
    model: torch.nn.Module,
    maps_cfg: Dict[str, dict],
    episodes_per_map: int = 20,
    batch_size: int = 32,
    replay_buffer_size: int = 1000,
    decay_range: int = 1000,
    lr: float = 1e-3,
    device: str = "cuda",
    log_dir: str = "logs",
    run_dir: Optional[str] = None,
    max_episode_seconds: int = 30,
    max_steps_cap: Optional[int] = None,  # 新增：步数上限（例如 2000），None 则只用动态回合上限
) -> pd.DataFrame:

    # 输出目录
    run_dir = Path(run_dir or (Path(log_dir) / get_timestamp()))
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    # 为了拿动作空间，先用第一张图构建一个 env
    first_name = next(iter(maps_cfg))
    first_env = build_env_from_raw(maps_cfg[first_name])

    agent = DDQNAgent(
        model,
        first_env.get_action_space(),
        lr=lr,
        decay_range=decay_range,
        device=device,
        replay_buffer_size=replay_buffer_size,
    )

    # 每图的统计容器
    results = []
    global_success_sum = 0
    global_ep_cnt = 0

    # 逐图评估
    for map_name, spec in maps_cfg.items():
        feats = extract_features_from_spec(map_name, spec)
        env = build_env_from_raw(spec)
        pbar = tqdm(range(episodes_per_map), desc=f"Map[{map_name}]", dynamic_ncols=True)

        success_cnt = 0
        timeout_cnt = 0
        for ep in pbar:
            out = run_one_episode(
                env, agent, ep,
                max_episode_seconds=max_episode_seconds,
                max_steps_cap=max_steps_cap
            )
            success_cnt += out["success"]
            timeout_cnt += out["timeout"]

            # 在线更新（可选：每若干步再训练一次）
            if len(agent.replay_buffer) >= batch_size:
                _ = agent.retrain(batch_size)

            # TB 逐集记录
            writer.add_scalar(f"{map_name}/success", out["success"], ep)
            writer.add_scalar(f"{map_name}/steps", out["steps"], ep)
            writer.add_scalar(f"{map_name}/timeout", out["timeout"], ep)

            pbar.set_postfix(sr=f"{success_cnt/(ep+1):.2f}")

            global_success_sum += out["success"]
            global_ep_cnt += 1
            writer.add_scalar("GLOBAL/success_rate_running", global_success_sum / max(1, global_ep_cnt), global_ep_cnt)

        sr = success_cnt / max(1, episodes_per_map)
        timeout_rate = timeout_cnt / max(1, episodes_per_map)

        row = dict(feats)
        row["success_rate"] = float(sr)
        row["timeout_rate"] = float(timeout_rate)
        results.append(row)

    df = pd.DataFrame(results)

    # 保存 CSV
    csv_path = run_dir / "features_vs_success.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    global_sr = df["success_rate"].mean() if len(df) else 0.0
    print(f"🌍 Global Success Rate (mean of per-map SR): {global_sr:.3f}")
    print(f"📝 结果已保存：{csv_path}")

    # 可视化
    try:
        make_feature_success_plots(run_dir, df)
        print(f"📊 可视化已输出到：{run_dir}")
    except Exception as e:
        print(f"⚠️ 生成可视化失败：{e}")

    writer.close()
    return df

# ============ 可视化：Features vs Success ============
def make_feature_success_plots(out_dir: Union[str, Path], df: pd.DataFrame):
    """
    生成：
    - success_rate by map 的柱状图
    - 每个特征 vs success_rate 的散点图（自动挑数值列）
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
        if c in ["map_name", "success_rate"]:
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

    # 3) 相关性热图（用 matplotlib，避免 seaborn 依赖）
    try:
        corr_cols = ["success_rate"] + numeric_cols
        dnum = df[corr_cols].copy()
        dnum = dnum.apply(pd.to_numeric, errors="coerce")
        cm = dnum.corr().values
        labels = dnum.columns.tolist()

        plt.figure(figsize=(1.2*len(labels), 1.0*len(labels)))
        im = plt.imshow(cm, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, fraction=0.046, pad=0.4)
        plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.title("Correlation (numeric features & success_rate)")
        # 在格子里标注相关系数
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "correlation_heatmap.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"[Plot] correlation_heatmap.png 失败: {e}")

# ============ 入口 ============
if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 读取你的 YAML（使用你给的路径）
    MAP_SETTINGS_PATH = r'C:/Users/MSc_SEIoT_1/MAPF_G2RL-main - train/g2rl/map_settings_generated_new.yaml'
    with open(MAP_SETTINGS_PATH, "r", encoding="utf-8") as f:
        base_map_settings = yaml.safe_load(f)

    # 顶层是 list → {name: spec}
    if isinstance(base_map_settings, list):
        base_map_settings = {
            (m.get("name") or f"map_{i}"): m
            for i, m in enumerate(base_map_settings)
        }

    # 模型 & 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNNModel().to(device)

    # 对每张图跑固定若干集，统计 features vs success
    df = train(
        model=model,
        maps_cfg=base_map_settings,
        episodes_per_map=300,                 # ⬅️ 每张图评估多少集（可调）
        batch_size=32,
        replay_buffer_size=1000,
        decay_range=1000,
        lr=1e-3,
        device=device,
        log_dir='logs',
        run_dir="C:/Users/MSc_SEIoT_1/MAPF_G2RL-main - train/features_success_run_1",
        max_episode_seconds=30,               # ⬅️ 墙钟超时（秒）
        max_steps_cap=2000,                   # ⬅️ 步数上限（None 表示不用）
    )

    # 可选：保存模型
    torch.save(model.state_dict(), 'models/best_model_1.pt')
    print('✅ 模型已保存到 models/best_model.pt')


