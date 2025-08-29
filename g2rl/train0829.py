#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union, List
from collections import deque
from types import SimpleNamespace
import os, sys, time, inspect, random

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm
from pogema import AStarAgent

# ===================== 配置区（按需修改） =====================
# 1) YAML 地图配置
MAP_SETTINGS_PATH = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main - train/g2rl/map_settings_generated_new.yaml"

# 2) 复杂度 CSV（由 infer_complexity.py 生成）
COMPLEXITY_CSV = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main-nn/0827result/maps_features_with_complexity.csv"

# 3) 复杂度CSV里用于匹配地图ID的列（csv）与 YAML 的对应字段（yaml）
MAP_ID_COL_IN_CSV  = "config_id"   # 也可换成 "grid_hash"
MAP_ID_COL_IN_YAML = "name"        # YAML 里地图的唯一标识字段

# 4) 训练超参
N_STAGES = 5
MIN_PER_STAGE = 10
MIN_EPISODES_PER_STAGE = 300
THRESHOLD = 0.80
WINDOW_SIZE = 100
NUM_EPISODES = 300
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 500
DECAY_RANGE = 10_000
MAX_EPISODE_SECONDS = 30
LOG_DIR = "logs"
RUN_DIR = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main/final_trainig_0829"
MODEL_OUT = "models/best_model_0829.pt"

# 防卡死：目标智能体连续多少步不动就提前终止该集
STUCK_PATIENCE = 20
# ============================================================

# ============ 项目根路径（确保能 import g2rl.*） ============
project_root = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main-nn"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============ 项目模块 ============
from g2rl.environment import G2RLEnv
from g2rl.agent import DDQNAgent
from g2rl.network import CRNNModel

# ---------------- 工具：从观测中提取数组 ----------------
def _find_array(obj):
    import numpy as np
    try:
        import torch
        is_tensor = True
    except Exception:
        is_tensor = False

    if isinstance(obj, np.ndarray):
        return obj
    if is_tensor and hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        try:
            return obj.detach().cpu().numpy()
        except Exception:
            pass
    if isinstance(obj, (list, tuple)):
        for v in obj:
            arr = _find_array(v)
            if arr is not None:
                return arr
        return None
    if isinstance(obj, dict):
        preferred = ("view_cache","obs","observation","state","tensor","grid","image","local_obs","global_obs")
        for k in preferred:
            if k in obj:
                arr = _find_array(obj[k])
                if arr is not None:
                    return arr
        for v in obj.values():
            arr = _find_array(v)
            if arr is not None:
                return arr
        return None
    try:
        arr = np.array(obj)
        if arr.ndim > 0:
            return arr
    except Exception:
        pass
    return None

# -------- 观测统一到 [1, C, D, H, W]，并对齐通道数 --------
def _obs_to_tensor_CDHW(s, device, expected_c: int):
    import numpy as np
    import torch
    arr = _find_array(s)
    if arr is None:
        raise ValueError("无法从观测中提取数组/张量。")
    arr = np.array(arr)

    # 归一到 [C, D, H, W]
    if arr.ndim == 5:  # [N,*,*,*,*]
        if arr.shape[1] == expected_c:
            arr = arr[0]
        elif arr.shape[-1] == expected_c:
            arr = np.transpose(arr, (0,4,1,2,3))[0]  # NDHWC -> CDHW
        else:
            arr = arr[0]
    elif arr.ndim == 4:  # [C,D,H,W] 或 [D,H,W,C] 或 [D,C,H,W]
        if arr.shape[0] == expected_c:
            pass
        elif arr.shape[-1] == expected_c:
            arr = np.transpose(arr, (3,0,1,2))       # DHWC -> CDHW
        elif arr.shape[1] == expected_c:
            arr = np.transpose(arr, (1,0,2,3))       # DCHW -> CDHW
    elif arr.ndim == 3:  # [D,H,W]
        arr = arr[None, ...]
    elif arr.ndim == 2:  # [H,W]
        arr = arr[None, None, ...]
    else:
        raise ValueError(f"观测维度不支持：shape={arr.shape}, ndim={arr.ndim}")

    # 对齐通道数
    C, D, H, W = arr.shape
    if C == expected_c:
        arr_fixed = arr
    elif C == 1 and expected_c > 1:
        arr_fixed = np.repeat(arr, expected_c, axis=0)
    elif C < expected_c:
        pad = np.zeros((expected_c - C, D, H, W), dtype=arr.dtype)
        arr_fixed = np.concatenate([arr, pad], axis=0)
    else:
        arr_fixed = arr[:expected_c]

    return torch.tensor(arr_fixed[None, ...], dtype=torch.float32, device=device)  # [1,C,D,H,W]

# -------- 稳健拿动作空间 --------
def _safe_get_action_space(env):
    asp = getattr(env, "action_space", None)
    if asp is not None and hasattr(asp, "n"):
        return asp
    gas = getattr(env, "get_action_space", None)
    if callable(gas):
        try:
            out = gas()
            if hasattr(out, "n"): return out
            if isinstance(out, (list, tuple)): return SimpleNamespace(n=len(out))
            if isinstance(out, int) and out > 0: return SimpleNamespace(n=out)
        except Exception:
            pass
    if hasattr(env, "actions"):
        try:
            return SimpleNamespace(n=len(env.actions))
        except Exception:
            pass
    # 重置再试一轮
    try:
        env.reset()
    except Exception:
        pass
    asp = getattr(env, "action_space", None)
    if asp is not None and hasattr(asp, "n"):
        return asp
    gas = getattr(env, "get_action_space", None)
    if callable(gas):
        try:
            out = gas()
            if hasattr(out, "n"): return out
            if isinstance(out, (list, tuple)): return SimpleNamespace(n=len(out))
            if isinstance(out, int) and out > 0: return SimpleNamespace(n=out)
        except Exception:
            pass
    if hasattr(env, "actions"):
        try:
            return SimpleNamespace(n=len(env.actions))
        except Exception:
            pass
    return SimpleNamespace(n=5)

# -------- 安全构造 env（只传 __init__ 支持的参数） --------
def build_env_from_raw(raw_cfg: dict) -> G2RLEnv:
    sig = inspect.signature(G2RLEnv.__init__)
    allowed = {
        p.name for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    allowed.discard("self")

    ctor_cfg = {k: v for k, v in raw_cfg.items() if k in allowed}
    env = G2RLEnv(**ctor_cfg)

    # 将 grid/starts/goals 等作为属性挂载（如果 YAML 有这些）
    if "grid" in raw_cfg:
        try:
            env.grid = (np.array(raw_cfg["grid"]) > 0).astype(np.uint8)
        except Exception:
            env.grid = None
    if "starts" in raw_cfg:
        env.starts = raw_cfg["starts"]
    if "goals" in raw_cfg:
        env.goals = raw_cfg["goals"]

    return env

# ================== 用 CSV 合并复杂度 ==================
def _merge_complexity_from_csv(base_map_settings: Dict[str, dict]) -> Dict[str, dict]:
    import numpy as np
    import pandas as pd
    import os

    if not os.path.exists(COMPLEXITY_CSV):
        raise FileNotFoundError(f"未找到复杂度CSV：{COMPLEXITY_CSV}")

    df = pd.read_csv(COMPLEXITY_CSV)

    if "nn_complexity" not in df.columns:
        if "nn_pred_success_rate" in df.columns:
            df["nn_complexity"] = 1.0 - pd.to_numeric(df["nn_pred_success_rate"], errors="coerce")
        else:
            raise ValueError("复杂度CSV缺少 nn_complexity 且没有 nn_pred_success_rate 无法推导。")

    df["nn_complexity"] = pd.to_numeric(df["nn_complexity"], errors="coerce")

    csv_id_candidates  = [c for c in ["config_id", "grid_hash", "name", "map_id"] if c in df.columns]
    yaml_id_candidates = set()
    for spec in base_map_settings.values():
        yaml_id_candidates.update(spec.keys())
    yaml_id_candidates = [c for c in ["config_id", "grid_hash", "name", "map_id"] if c in yaml_id_candidates]

    common_ids = [c for c in csv_id_candidates if c in yaml_id_candidates]
    chosen_id = common_ids[0] if common_ids else None
    use_composite = (chosen_id is None)

    if not use_composite:
        agg = (
            df.groupby(chosen_id, dropna=False)["nn_complexity"]
              .mean().reset_index().rename(columns={"nn_complexity": "complexity"})
        )
        comp_map = dict(zip(agg[chosen_id].astype(str), agg["complexity"].astype(float)))
    else:
        needed = ["size", "num_agents", "density", "obs_radius", "max_episode_steps"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"无法用复合键匹配，CSV 缺少列：{missing}")

        df["_size"] = pd.to_numeric(df["size"], errors="coerce").astype("Int64")
        df["_nag"]  = pd.to_numeric(df["num_agents"], errors="coerce").astype("Int64")
        df["_den"]  = pd.to_numeric(df["density"], errors="coerce").round(4)
        df["_obs"]  = pd.to_numeric(df["obs_radius"], errors="coerce").astype("Int64")
        df["_mep"]  = pd.to_numeric(df["max_episode_steps"], errors="coerce").astype("Int64")

        comp = (
            df.dropna(subset=["_size","_nag","_den","_obs","_mep"])
              .groupby(["_size","_nag","_den","_obs","_mep"])["nn_complexity"]
              .mean().reset_index().rename(columns={"nn_complexity":"complexity"})
        )
        comp_map = { (int(r["_size"]), int(r["_nag"]), float(r["_den"]), int(r["_obs"]), int(r["_mep"])): float(r["complexity"])
                     for _, r in comp.iterrows() }

    matched, unmatched = 0, 0
    out: Dict[str, dict] = {}

    for name, spec in base_map_settings.items():
        new_spec = dict(spec)
        cpx_val = np.nan

        if not use_composite:
            key = spec.get(chosen_id, None)
            if key is None and chosen_id == "name":
                key = name
            if key is not None:
                cpx_val = comp_map.get(str(key), np.nan)
        else:
            try:
                size  = int(spec.get("size"))
                nag   = int(spec.get("num_agents"))
                den   = float(spec.get("density"))
                obs   = int(spec.get("obs_radius"))
                mep   = int(spec.get("max_episode_steps"))
                tup   = (size, nag, round(den,4), obs, mep)
                cpx_val = comp_map.get(tup, np.nan)
            except Exception:
                cpx_val = np.nan

        if np.isfinite(cpx_val):
            matched += 1
            new_spec["complexity"] = float(cpx_val)
        else:
            unmatched += 1
            new_spec["complexity"] = np.nan

        out[name] = new_spec

    print(f"🧩 复杂度匹配统计：matched={matched}, unmatched={unmatched} | 模式={'复合键' if use_composite else f'ID列({chosen_id})'}")
    if unmatched > 0:
        examples = [k for k,v in out.items() if not np.isfinite(v.get('complexity', np.nan))][:5]
        print(f"⚠️ 有 {unmatched} 张地图未匹配到复杂度（未参与量化）。示例：{examples}")
    return out

# ================== 阶段切分（按复杂度分位） ==================
def _build_stages_by_quantile_df(df: pd.DataFrame, n_stages: int = 5, min_per_stage: int = 5):
    df = df.dropna(subset=["complexity"]).copy()
    if len(df) == 0:
        raise ValueError("没有可用地图用于 complexity 课程（complexity 全 NaN）。")
    qs = np.linspace(0, 1, n_stages + 1)
    edges = np.quantile(df["complexity"].values, qs)
    stages = []
    for i in range(n_stages):
        lo, hi = float(edges[i]), float(edges[i+1]) + 1e-12
        sub = df[(df["complexity"] >= lo) & (df["complexity"] < hi)]
        if len(sub) < min_per_stage:
            need = min_per_stage - len(sub)
            center = (lo + hi) / 2.0
            extra = df.iloc[(df["complexity"] - center).abs().argsort()[:need]]
            sub = pd.concat([sub, extra]).drop_duplicates(subset=["name"])
        stages.append({
            "stage": i,
            "cpx_min": lo,
            "cpx_max": hi,
            "items": sub.to_dict("records"),
        })
    return stages

# ================== Scheduler ==================
class ComplexityScheduler:
    """
    - 每阶段持有一组 maps（由 complexity 分位切分）
    - 每阶段至少跑 min_episodes_per_stage 集
    - 达到阈值后晋级
    """
    def __init__(self,
                 base_map_settings: Dict[str, dict],
                 n_stages: int = 5,
                 min_per_stage: int = 10,
                 min_episodes_per_stage: int = 100,
                 threshold: float = 0.70,
                 window_size: int = 100,
                 use_window_sr: bool = True,
                 shuffle_each_stage: bool = True,
                 seed: int = 0):
        self.min_episodes_per_stage = int(min_episodes_per_stage)
        self.threshold = float(threshold)
        self.window_size = int(window_size)
        self.use_window_sr = bool(use_window_sr)

        rows = []
        for name, spec in base_map_settings.items():
            rows.append({"name": name, "complexity": spec.get("complexity", np.nan), "spec": spec})
        df = pd.DataFrame(rows)
        stages = _build_stages_by_quantile_df(df, n_stages=n_stages, min_per_stage=min_per_stage)

        self._rng = random.Random(seed)
        self._stage_items = []
        self._stage_edges = []
        for st in stages:
            items = list(st["items"])
            if shuffle_each_stage:
                self._rng.shuffle(items)
            self._stage_items.append(items)
            self._stage_edges.append((st["cpx_min"], st["cpx_max"]))

        self.current_stage = 0
        self.max_stage = len(self._stage_items) - 1

        self._idx_in_stage = 0
        self._win = deque(maxlen=self.window_size)
        self._ep_in_stage = 0
        self._succ_in_stage = 0

    def get_updated_map_settings(self) -> Dict[str, dict]:
        if self.current_stage > self.max_stage:
            return {}
        items = self._stage_items[self.current_stage]
        if not items:
            raise RuntimeError(f"Stage {self.current_stage} 没有地图。")
        item = items[self._idx_in_stage]
        self._idx_in_stage = (self._idx_in_stage + 1) % len(items)
        return {item["name"]: item["spec"]}

    def add_episode_result(self, success: int):
        s = 1 if success else 0
        self._win.append(s)
        self._ep_in_stage += 1
        self._succ_in_stage += s

    def window_sr(self) -> float:
        return float(sum(self._win) / len(self._win)) if len(self._win) else 0.0

    def stage_sr(self) -> float:
        return float(self._succ_in_stage / max(1, self._ep_in_stage))

    def should_advance(self) -> bool:
        if self._ep_in_stage < self.min_episodes_per_stage:
            return False
        sr = self.window_sr() if self.use_window_sr else self.stage_sr()
        return sr >= self.threshold

    def advance(self, pbar=None):
        if pbar:
            lo, hi = self._stage_edges[self.current_stage]
            pbar.write(
                f"✅ 通过 Stage {self.current_stage} | "
                f"SR(win)={self.window_sr():.2f} / SR(stage)={self.stage_sr():.2f} | "
                f"区间[{lo:.4f}, {hi:.4f}] → Stage {self.current_stage + 1}"
            )
        self.current_stage += 1
        self._reset_stage_stats()

    def repeat_stage(self, pbar=None):
        if pbar:
            pbar.write(f"🔁 未达标，重复 Stage {self.current_stage}（已训练 {self._ep_in_stage} ep，SR={self.stage_sr():.2f}）")
        self._reset_stage_stats()

    def _reset_stage_stats(self):
        self._idx_in_stage = 0
        self._win.clear()
        self._ep_in_stage = 0
        self._succ_in_stage = 0

    def is_done(self) -> bool:
        return self.current_stage > self.max_stage

# ================== 训练 ==================
def get_timestamp() -> str:
    return datetime.now().strftime('%H-%M-%d-%m-%Y')

def train(
    model: torch.nn.Module,
    map_settings: Dict[str, dict],
    map_probs: Union[List[float], None],
    num_episodes: int = 300,
    batch_size: int = 32,
    decay_range: int = 1000,
    log_dir='logs',
    lr: float = 0.001,
    replay_buffer_size: int = 1000,
    device: str = 'cuda',
    scheduler: Optional[ComplexityScheduler] = None,
    max_episode_seconds: int = 30,
    run_dir: Optional[str] = None
) -> DDQNAgent:

    # 输出目录
    run_dir = Path(log_dir) / get_timestamp() if run_dir is None else Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    # 初始化第一个 env 以获取动作空间
    first_name = next(iter(map_settings))
    first_env = build_env_from_raw(map_settings[first_name])
    try:
        first_env.reset()
    except Exception:
        pass

    asp = _safe_get_action_space(first_env)
    n_actions = int(getattr(asp, "n", 5))
    action_space_list = list(range(n_actions))
    print(f"✅ n_actions = {n_actions}")

    # 探针观测 & 预热 LazyConv3d
    try:
        obs0, _ = first_env.reset()
    except Exception:
        obs0 = first_env.reset()
    state0 = obs0[0] if isinstance(obs0, (list, tuple)) else obs0
    in_channels = 11  # 你的输入通道数（常见 11）
    with torch.no_grad():
        _ = model(_obs_to_tensor_CDHW(state0, device, expected_c=in_channels))
    print(f"🔥 warmed model with in_channels={in_channels}")

    # 单样本预处理（存经验/拼 batch 用）：[C,D,H,W]
    def preprocess_single(s):
        return _obs_to_tensor_CDHW(s, device, expected_c=in_channels).squeeze(0)

    # 初始化 Agent（兼容两种构造签名）
    try:
        agent = DDQNAgent(
            model,                # q_network
            model,                # model
            action_space_list,
            lr=lr,
            decay_range=decay_range,
            device=device,
            replay_buffer_size=replay_buffer_size,
            obs_preprocessor=preprocess_single,
        )
    except TypeError:
        agent = DDQNAgent(
            model,
            action_space_list,
            lr=lr,
            decay_range=decay_range,
            device=device,
            replay_buffer_size=replay_buffer_size,
            obs_preprocessor=preprocess_single,
        )

    training_logs = []
    pbar = tqdm(range(num_episodes), desc='Episodes', dynamic_ncols=True)

    episode = 0
    success_count_total = 0

    while (scheduler is None) or (scheduler.current_stage <= scheduler.max_stage):
        if episode >= num_episodes:
            break

        # 1) 当前阶段地图
        cur_map_cfg = scheduler.get_updated_map_settings() if scheduler else map_settings
        map_type, cfg = next(iter(cur_map_cfg.items()))
        env = build_env_from_raw(cfg)

        stage_id = (scheduler.current_stage if scheduler else -1)
        cpx_val = cfg.get("complexity", None)
        if cpx_val is not None and np.isfinite(cpx_val):
            pbar.write(f"🟢 地图：{map_type} | Stage {stage_id} | Agents={env.num_agents} | Complexity={float(cpx_val):.3f}")
        else:
            pbar.write(f"🟢 地图：{map_type} | Stage {stage_id} | Agents={env.num_agents}")

        # 2) reset
        try:
            obs, info = env.reset()
        except Exception:
            obs = env.reset()
            info = {}

        # 目标代理（其它用 A* 做队友）
        target_idx = np.random.randint(env.num_agents)
        teammates = [AStarAgent() if i != target_idx else None for i in range(env.num_agents)]
        goal = tuple(env.goals[target_idx])
        state = obs[target_idx]

        # 估计最短路长度（或用曼哈顿作兜底）→ 动态步数预算
        try:
            opt_len = max(1, len(env.global_guidance[target_idx]) + 1)
        except Exception:
            try:
                sx, sy = state['global_xy']; gx, gy = goal
                opt_len = max(1, abs(sx - gx) + abs(sy - gy) + 1)
            except Exception:
                opt_len = 60
        timesteps_per_episode = min(400, max(60, int(opt_len * 6)))

        # 3) 一集
        success_flag = False
        retrain_count = 0
        episode_start_time = time.time()

        # 防卡死
        last_pos = tuple(state['global_xy'])
        no_move_steps = 0

        for t in range(timesteps_per_episode):
            if time.time() - episode_start_time > max_episode_seconds:
                pbar.write(f"⏰ Episode {episode} 超时（>{max_episode_seconds}s）")
                break

            # ===== 我方（NN） + 队友（A*） 选动作 =====
            actions = []
            for i in range(env.num_agents):
                if i == target_idx:
                    x = _obs_to_tensor_CDHW(obs[i], device, expected_c=in_channels)  # [1,C,D,H,W]
                    if x.shape[1] != in_channels:
                        raise RuntimeError(f"[PrepMismatch] x.shape={tuple(x.shape)}, expected C={in_channels}")
                    with torch.no_grad():
                        q = agent.q_network(x)
                        a = int(torch.argmax(q, dim=1).item())
                    actions.append(a)
                else:
                    try:
                        actions.append(int(teammates[i].act(obs[i])))
                    except Exception:
                        actions.append(0)

            obs, reward, terminated, truncated, info = env.step(actions)

            # 成功判定
            agent_pos = tuple(obs[target_idx]['global_xy'])
            done = (agent_pos == goal)
            terminated[target_idx] = done
            if done:
                success_flag = True
                break

            # 防卡死：若目标 agent 一直没移动，提前终止
            if agent_pos == last_pos:
                no_move_steps += 1
                if no_move_steps >= STUCK_PATIENCE:
                    pbar.write(f"🧊 Episode {episode} 提前终止（{STUCK_PATIENCE} 步未移动）")
                    break
            else:
                no_move_steps = 0
                last_pos = agent_pos

            # 经验 & 学习
            agent.store(
                state,
                actions[target_idx],
                reward[target_idx],
                obs[target_idx],
                terminated[target_idx],
            )
            state = obs[target_idx]

            if len(agent.replay_buffer) >= batch_size:
                retrain_count += 1
                _ = agent.retrain(batch_size)

        # 4) 统计与日志
        if success_flag:
            success_count_total += 1

        episode += 1
        success_rate = success_count_total / max(1, episode)
        writer.add_scalar('Success/episode', int(success_flag), episode)
        writer.add_scalar('SuccessRate/global', success_rate, episode)

        pbar.set_postfix(
            Stage=(stage_id if scheduler else "-"),
            success=int(success_flag),
            SR=f"{success_rate:.3f}",
        )
        pbar.update(1)

        training_logs.append({
            "episode": episode,
            "stage": stage_id,
            "map": map_type,
            "agents": getattr(env, "num_agents", None),
            "complexity": (float(cpx_val) if cpx_val is not None else np.nan),
            "success": int(success_flag),
            "success_rate": float(success_rate),
        })

        # 5) 课程逻辑
        if scheduler is not None:
            scheduler.add_episode_result(int(success_flag))
            if scheduler.should_advance():
                scheduler.advance(pbar)
                if scheduler.is_done():
                    break
            else:
                if scheduler._ep_in_stage >= scheduler.min_episodes_per_stage:
                    scheduler.repeat_stage(pbar)

    # 保存日志
    df = pd.DataFrame(training_logs)
    csv_path = run_dir / "episodes.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"📝 每集日志已保存：{csv_path}")

    writer.close()
    return agent

# ================== 主程序 ==================
if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 1) 读 YAML
    with open(MAP_SETTINGS_PATH, "r", encoding="utf-8") as f:
        base_map_settings = yaml.safe_load(f)

    if isinstance(base_map_settings, list):
        base_map_settings = {(m.get("name") or f"map_{i}"): m for i, m in enumerate(base_map_settings)}

    # 2) 合并复杂度（来自 CSV 的 nn_complexity）
    base_map_settings = _merge_complexity_from_csv(base_map_settings)

    # 3) 构建 Scheduler
    scheduler = ComplexityScheduler(
        base_map_settings=base_map_settings,
        n_stages=N_STAGES,
        min_per_stage=MIN_PER_STAGE,
        min_episodes_per_stage=MIN_EPISODES_PER_STAGE,
        threshold=THRESHOLD,
        window_size=WINDOW_SIZE,
        use_window_sr=True,
        shuffle_each_stage=True,
        seed=0,
    )

    # 4) 设备 & 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNNModel().to(device)

    # 5) 训练
    agent = train(
        model=model,
        scheduler=scheduler,
        map_settings=scheduler.get_updated_map_settings(),  # 初次取一张；train 内每集会再取
        map_probs=None,
        num_episodes=NUM_EPISODES,
        batch_size=BATCH_SIZE,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        decay_range=DECAY_RANGE,
        log_dir=LOG_DIR,
        device=device,
        max_episode_seconds=MAX_EPISODE_SECONDS,
        run_dir=RUN_DIR,
    )

    # 6) 保存模型权重
    out_path = Path(MODEL_OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path.as_posix())
    print(f"✅ 模型已保存到 {out_path}")
