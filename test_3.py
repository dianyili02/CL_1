# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-

import os
import sys
import math
import time
import argparse
import random
import inspect
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch

# ---------- Project root ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------- Project imports ----------
from g2rl.environment import G2RLEnv
from g2rl.agent import DDQNAgent
from g2rl.network import CRNNModel
from g2rl.complexity import compute_complexity  # 仅用来取 LDD/BN/MC/DLR
from pogema import AStarAgent


# ====================== 更新的“z-score”复杂度公式 ======================
NEW_INTERCEPT = 0.8498563938534148
Z_COEFFS = {
    "size":           -0.053114841431693365,
    "num_agents":     +0.2522555834547447,
    "density":        -0.024373645110113036,
    "density_actual": -0.022860995775357107,
    "LDD":            -0.07455248304868833,
    "BN":             +0.04608927828640109,
    "MC":             +0.08999452177843661,
    "DLR":            +0.07483901107179369,
    "FRA":            +0.04616983038496823,
    "FPA":            +0.05103388422320293,
}
Z_FEATURE_KEYS = list(Z_COEFFS.keys())

# ------------------------- Data structures -------------------------
@dataclass
class MapConfig:
    size: int
    num_agents: int
    density: float
    obs_radius: int = 5
    max_episode_steps: int = 100
    seed: Optional[int] = None
    max_episode_seconds: int = 30  # wall-clock safety

@dataclass

class MapRecord:
    map_id: str
    size: int
    agents: int
    density: float
    seed: int
    density_actual: Optional[float] = None
    LDD: Optional[float] = None
    BN: Optional[float] = None
    MC: Optional[float] = None
    DLR: Optional[float] = None
    FPA: Optional[float] = None   # 新增
    FRA: Optional[float] = None   # 新增
    complexity: Optional[float] = None


# ------------------------- Core env/utils -------------------------
def _env_num_agents(env) -> int:
    gc = getattr(env, "grid_config", None)
    if gc is not None and hasattr(gc, "num_agents"):
        try:
            return int(gc.num_agents)
        except Exception:
            pass
    v = getattr(env, "num_agents", None)
    return int(v) if isinstance(v, int) and v > 0 else 1

def _infer_num_actions(env) -> int:
    # 优先：和 G2RLEnv.step 用的 actions 对齐
    actions = getattr(env, "actions", None)
    if isinstance(actions, (list, tuple)) and len(actions) > 0:
        return len(actions)

    for name in ["action_space_n", "n_actions", "num_actions"]:
        v = getattr(env, name, None)
        if isinstance(v, int) and v > 0:
            return v

    asp = getattr(env, "action_space", None)
    if asp is not None and hasattr(asp, "n"):
        return int(asp.n)

    gas = getattr(env, "get_action_space", None)
    if callable(gas):
        try:
            obj = gas()
            if hasattr(obj, "n"):
                return int(obj.n)
            if isinstance(obj, int) and obj > 0:
                return obj
        except Exception:
            pass

    # 兜底：和上面 _make_env_from_config 的默认动作长度一致
    return 5


def _get_action_space(env):
    asp = getattr(env, "action_space", None)
    if asp is not None:
        return asp
    gas = getattr(env, "get_action_space", None)
    if callable(gas):
        try:
            return gas()
        except Exception:
            pass
    from types import SimpleNamespace
    return SimpleNamespace(n=_infer_num_actions(env))

def _to_scalar_per_agent(x, target_idx):
    if isinstance(x, dict):
        if target_idx in x:
            return x[target_idx]
        for k in ["global", "all", "any", "done", "terminated"]:
            if k in x:
                return x[k]
        try:
            return next(iter(x.values()))
        except Exception:
            return 0
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return 0
        if target_idx < len(x):
            return x[target_idx]
        try:
            return bool(np.any(x))
        except Exception:
            return x[0]
    return x

def _env_step_joint(env, joint_actions, target_idx):
    out = env.step(joint_actions)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
    elif isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        terminated, truncated = done, False
    else:
        obs, reward, terminated, truncated, info = out

    reward_s     = _to_scalar_per_agent(reward,     target_idx)
    terminated_s = bool(_to_scalar_per_agent(terminated, target_idx))
    truncated_s  = bool(_to_scalar_per_agent(truncated,  target_idx))
    return obs, reward_s, terminated_s, truncated_s, info

def _extract_grid(env) -> Optional[np.ndarray]:
    def _is_grid(a):
        try:
            arr = np.array(a); return arr.ndim == 2
        except Exception:
            return False
    for name in ["grid", "grid_map", "grid_matrix", "grid_array", "occupancy", "occupancy_grid", "obstacle_map", "map", "matrix"]:
        a = getattr(env, name, None)
        if _is_grid(a): arr = np.array(a); return (arr > 0).astype(np.uint8)
    gc = getattr(env, "grid_config", None)
    if gc is not None:
        for name in ["grid", "grid_map", "grid_matrix", "grid_array", "occupancy", "occupancy_grid", "obstacle_map", "map", "matrix"]:
            a = getattr(gc, name, None)
            if _is_grid(a): arr = np.array(a); return (arr > 0).astype(np.uint8)
    return None

def _get_pos_goal_from_env_and_state(env, state, target_idx):
    pos = goal = None
    try:
        if hasattr(env, "goals"):
            g = env.goals[target_idx]; goal = (int(g[0]), int(g[1]))
    except Exception: pass
    try:
        if hasattr(env, "starts"):
            s = env.starts[target_idx]; pos = (int(s[0]), int(s[1]))
    except Exception: pass
    if isinstance(state, dict):
        if pos is None:
            gx = state.get("global_xy") or state.get("pos")
            if gx is not None:
                try: pos = (int(gx[0]), int(gx[1]))
                except Exception: pass
        if goal is None:
            gt = state.get("global_target_xy") or state.get("goal")
            if gt is not None:
                try: goal = (int(gt[0]), int(gt[1]))
                except Exception: pass
    return pos, goal

def _estimate_opt_len(env, state, target_idx, start_pos, goal_pos):
    try:
        path = [state["global_xy"]] + env.global_guidance[target_idx]
        return max(1, len(path) - 1)
    except Exception:
        pass
    if start_pos is not None and goal_pos is not None:
        manhattan = abs(start_pos[0] - goal_pos[0]) + abs(start_pos[1] - goal_pos[1])
        return max(1, int(manhattan))
    return 1

# ------------------------- Build envs & sampling -------------------------
def _make_env_from_config(cfg: MapConfig) -> G2RLEnv:
    env = G2RLEnv(
        size=cfg.size,
        num_agents=cfg.num_agents,
        density=cfg.density,
        obs_radius=cfg.obs_radius,
        max_episode_steps=cfg.max_episode_steps,
        seed=cfg.seed,
    )

    # >>> 兜底：确保有 actions（G2RLEnv.step 会用 self.actions[action]）
    if not hasattr(env, "actions") or env.actions is None or len(env.actions) == 0:
        # 与 environment.py 中使用到的字符串保持一致：'idle' 而不是 'stay'
        env.actions = ['idle', 'up', 'down', 'left', 'right']

    # 可选：给个 n_actions，某些代码会读它
    if not hasattr(env, "n_actions"):
        env.n_actions = len(env.actions)

    # 传入墙钟超时参数
    if not hasattr(env, "_args"):
        setattr(env, "_args", {})
    env._args["max_episode_seconds"] = int(cfg.max_episode_seconds)
    return env


def _reset_env_with_seed_and_grid(env: G2RLEnv, seed: int) -> Tuple[Any, Dict]:
    try:
        return env.reset(seed=seed)
    except TypeError:
        obs = env.reset()
        return (obs, {})

def _extract_grid_or_synth(env, size, density, seed):
    try:
        grid = _extract_grid(env)
        if isinstance(grid, np.ndarray) and grid.ndim == 2:
            return grid
        raise ValueError("invalid grid")
    except Exception:
        rng = np.random.RandomState(seed)
        return (rng.rand(size, size) < float(density)).astype(np.uint8)

# =============== 新增：统计特征 → z-score → 预测复杂度 =================
def compute_feature_stats(records: List[MapRecord]) -> Dict[str, Tuple[float, float]]:
    """
    计算每个 z 特征的 (mean, std)；std 为 0 或 NaN 时设为 1，避免除零。
    """
    stats = {}
    # 组装 DataFrame 方便处理
    rows = []
    for r in records:
        rows.append({
    "size": r.size,
    "num_agents": r.agents,
    "density": r.density,
    "density_actual": r.density_actual,
    "LDD": r.LDD,
    "BN": r.BN,
    "MC": r.MC,
    "DLR": r.DLR,
    "FPA": r.FPA,
    "FRA": r.FRA,
})
    df = pd.DataFrame(rows)
    for k in Z_FEATURE_KEYS:
        if k in df.columns:
            series = pd.to_numeric(df[k], errors="coerce")
            mu = float(series.mean(skipna=True))
            sd = float(series.std(ddof=0, skipna=True))
            if not np.isfinite(sd) or sd == 0.0:
                sd = 1.0
            stats[k] = (mu, sd)
        else:
            stats[k] = (0.0, 1.0)
    return stats

def _z(v, mu_sd: Tuple[float, float]) -> float:
    mu, sd = mu_sd
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 0.0
        return float((v - mu) / sd)
    except Exception:
        return 0.0

def predict_complexity_z(rec: MapRecord, stats: Dict[str, Tuple[float, float]]) -> float:
    """
    按你的 z 公式计算复杂度；缺失值按 z=0 处理（即不贡献）。
    """
    feats = {
    "size":           rec.size,
    "num_agents":     rec.agents,
    "density":        rec.density,
    "density_actual": rec.density_actual,
    "LDD":            rec.LDD,
    "BN":             rec.BN,
    "MC":             rec.MC,
    "DLR":            rec.DLR,
    "FPA":            rec.FPA,
    "FRA":            rec.FRA,
}

    yhat = NEW_INTERCEPT
    for k, w in Z_COEFFS.items():
        zval = _z(feats.get(k, None), stats.get(k, (0.0, 1.0)))
        yhat += w * zval
    return float(yhat)

# ------------------------- Sampling maps -------------------------
def sample_grid_maps() -> List[MapRecord]:
    """
    生成一批地图，抽取 density_actual 与 LDD/BN/MC/DLR（尽量从 compute_map_complexity 获得）。
    复杂度先不算，后面根据 z 统计统一计算。
    """
    map_sizes    = [32, 64, 96, 128]
    agent_counts = [2, 4, 8, 16]
    densities    = [0.10, 0.30, 0.50, 0.70]

    records: List[MapRecord] = []
    idx = 0
    for size in map_sizes:
        for agents in agent_counts:
            for density in densities:
                seed = 1000 + idx
                idx += 1
                try:
                    cfg = MapConfig(size=size, num_agents=agents, density=density,
                                    obs_radius=5, max_episode_steps=100, seed=seed,
                                    max_episode_seconds=30)
                    env = _make_env_from_config(cfg)
                except Exception as e:
                    print(f"[ENV-ERROR] build env failed s={size} a={agents} d={density:.2f}: {e}")
                    # 回退：拿不到 env 就用合成 grid，仅能算 density_actual，其余 NaN
                    grid = (np.random.RandomState(seed).rand(size, size) < density).astype(np.uint8)
                    density_actual = float(grid.mean())
                    records.append(MapRecord(
                        map_id=f"grid_{idx}", size=size, agents=agents, density=density, seed=seed,
                        density_actual=density_actual, LDD=np.nan, BN=np.nan, MC=np.nan, DLR=np.nan,
                        complexity=None
                    ))
                    continue

                # reset
                try:
                    _reset_env_with_seed_and_grid(env, seed)
                except Exception as e:
                    print(f"[ENV-WARN] reset(seed) failed: {e}")
                    try: env.reset()
                    except Exception as e2: print(f"[ENV-ERROR] reset() failed: {e2}")

                # grid & density_actual
                grid = _extract_grid_or_synth(env, size, density, seed)
                density_actual = float(np.array(grid).mean())

                # 只从 compute_map_complexity 拿 LDD/BN/MC/DLR（拿不到则 NaN）
                LDD = BN = MC = DLR = np.nan
                FPA = FRA = np.nan   # 🔑 给默认值，避免 UnboundLocalError
                try:
                    comp = compute_complexity(grid)  # 你的模块：返回结构可能是 dict / tuple
                    # 宽松解析：
                    if isinstance(comp, dict):
                        LDD = float(comp.get("LDD", np.nan))
                        BN  = float(comp.get("BN",  np.nan))
                        MC  = float(comp.get("MC",  np.nan))
                        DLR = float(comp.get("DLR", np.nan))
                        FPA = float(comp.get("FPA", np.nan))  # 新增
                        FRA = float(comp.get("FRA", np.nan))
                    elif isinstance(comp, (list, tuple)) and len(comp) >= 2 and isinstance(comp[1], dict):
                        sub = comp[1]
                        LDD = float(comp.get("LDD", np.nan))
                        BN  = float(comp.get("BN",  np.nan))
                        MC  = float(comp.get("MC",  np.nan))
                        DLR = float(comp.get("DLR", np.nan))
                        FPA = float(comp.get("FPA", np.nan))  # 新增
                        FRA = float(comp.get("FRA", np.nan))
                except Exception as e:
                    # 忽略，保持 NaN
                    pass

                records.append(MapRecord(
    map_id=f"grid_{idx}", size=size, agents=agents, density=density, seed=seed,
    density_actual=density_actual, LDD=LDD, BN=BN, MC=MC, DLR=DLR,
    FPA=FPA, FRA=FRA, complexity=None
))


    return records

def split_into_stages(records: List[MapRecord], stages: int, max_complexity: Optional[float]) -> List[List[MapRecord]]:
    # 可选过滤
    if max_complexity is not None:
        records = [r for r in records
                   if (r.complexity is not None and np.isfinite(r.complexity) and r.complexity <= max_complexity)]
    # 按复杂度排序
    # 在 split_into_stages() 内把排序改为降序
    records = sorted(
        [r for r in records if r.complexity is not None and np.isfinite(r.complexity)],
        key=lambda r: r.complexity,
        reverse=False  # ← 改这里：从大到小
    )

    if not records:
        return []
    chunk = max(1, math.ceil(len(records) / stages))
    return [records[i:i+chunk] for i in range(0, len(records), chunk)]

# ------------------------- Action helpers -------------------------
_CALIB_CACHE = {}
_CALIB_CACHE = {}

def calibrate_action_encoding(env, target_idx: int):
    """
    返回 {'noop': idx_noop, 'up': idx_up, 'down': idx_down, 'left': idx_left, 'right': idx_right}
    优先使用 env.actions 的名称；若没有则回退到常用顺序。
    """
    key = id(env)
    if key in _CALIB_CACHE:
        return _CALIB_CACHE[key]

    names = getattr(env, "actions", None)
    mapping = None

    if isinstance(names, (list, tuple)) and len(names) > 0:
        name_to_idx = {str(n).lower(): i for i, n in enumerate(names)}
        # 兼容不同命名：idle/stay；right/rt 等等可以按需再加
        noop_idx = name_to_idx.get('idle', name_to_idx.get('stay', 0))
        mapping = {
            'noop':  noop_idx,
            'up':    name_to_idx.get('up', 1),
            'down':  name_to_idx.get('down', 2),
            'left':  name_to_idx.get('left', 3),
            'right': name_to_idx.get('right', 4),
        }
    else:
        # 回退到常用定义
        mapping = {'noop': 0, 'up': 1, 'down': 2, 'left': 3, 'right': 4}

    # 打印一次，便于确认
    effects = {mapping['noop']:(0,0), mapping['up']:(-1,0), mapping['down']:(1,0),
               mapping['left']:(0,-1), mapping['right']:(0,1)}
    print(f"[Calibrate] actions={getattr(env,'actions',None)}  mapping={mapping} effects={effects}")

    _CALIB_CACHE[key] = mapping
    return mapping


def toward_with_mapping(dx: int, dy: int, mapping: Dict[str, int]) -> int:
    best = None
    for name, vec in {"up":(-1,0), "down":(1,0), "left":(0,-1), "right":(0,1)}.items():
        score = -(dx*vec[0] + dy*vec[1])
        if best is None or score < best[0]:
            best = (score, name)
    return mapping.get(best[1], mapping["noop"])

def _get_q_module(agent):
    for name in ("model", "net", "q_network", "qnet", "policy_net", "online_net", "network"):
        m = getattr(agent, name, None)
        if m is not None and hasattr(m, "parameters"):
            return m
    return None

# ------------------------- Episode rollout -------------------------
def _choose_target_idx(env, obs) -> int:
    num_agents = _env_num_agents(env)
    gl = getattr(env, "global_guidance", None)
    grid_arr = None
    try:
        grid_arr = _extract_grid(env)
    except Exception:
        pass

    def _free_degree(grid_arr, xy):
        if grid_arr is None or xy is None:
            return 4
        H, W = grid_arr.shape
        x, y = int(xy[0]), int(xy[1])
        deg = 0
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < H and 0 <= ny < W and grid_arr[nx, ny] == 0:
                deg += 1
        return deg

    if gl is not None:
        cands: List[Tuple[int,int]] = []
        for i in range(num_agents):
            try:
                L = len(gl[i])
                s_i = obs[i].get("global_xy", None) if isinstance(obs, (list, tuple)) else obs.get("global_xy", None)
                if L > 0 and _free_degree(grid_arr, s_i) >= 2:
                    cands.append((L, i))
            except Exception:
                pass
        if cands:
            cands.sort()
            return cands[0][1]
        best = None
        for i in range(num_agents):
            try:
                L = len(gl[i])
                if L > 0 and (best is None or L < best[0]):
                    best = (L, i)
            except Exception:
                pass
        if best is not None:
            return best[1]
    return np.random.randint(num_agents)

def _rollout_one_episode(env, agent, target_idx: Optional[int] = None, teammates: str = "astar",
                         steps_factor: float = 5.0, secs_factor: float = 1.0,
                         assist: str = "none",
                         assist_prob: float = 1.0,
                         assist_cooldown: int = 0,
                         assist_max_ratio: float = 1.0,
                         assist_warmup: int = 0,
                         assist_near_goal: int = 0) -> Dict[str, Any]:

    try:
        obs, info = env.reset(seed=getattr(env, "seed", None))
    except TypeError:
        obs = env.reset()
        info = {}

    num_agents = _env_num_agents(env)
    if target_idx is None:
        target_idx = _choose_target_idx(env, obs)

    mapping = calibrate_action_encoding(env, target_idx)
    noop = mapping["noop"]
    n_actions = _infer_num_actions(env)

    def _state_of(o, ti):
        try:
            return o[ti]
        except Exception:
            return o

    state = _state_of(obs, target_idx)
    start_pos, goal_pos = _get_pos_goal_from_env_and_state(env, state, target_idx)
    last_pos = start_pos

    opt_len = _estimate_opt_len(env, state, target_idx, start_pos, goal_pos)
    base_steps = int(getattr(env, "max_episode_steps", None)
                     or getattr(getattr(env, "grid_config", env), "max_episode_steps", None)
                     or (getattr(env, "size", 32) * 3))
    base_timeout = int(getattr(env, "_args", {}).get("max_episode_seconds", 30))
    max_steps = max(base_steps, int(opt_len * steps_factor))
    timeout_s = max(base_timeout, int(opt_len * secs_factor))

    density = getattr(env, "density", getattr(getattr(env, "grid_config", object), "density", 0.0))
    agents  = getattr(env, "num_agents", getattr(getattr(env, "grid_config", object), "num_agents", 1))
    if (float(density) >= 0.50) or (int(agents) >= 12):
        max_steps = max(max_steps, int(opt_len * max(steps_factor, 2.0)))
        timeout_s = max(timeout_s, int(opt_len * max(secs_factor,  2.0)))

    astar_team = None
    if teammates == "astar":
        astar_team = [AStarAgent() if i != target_idx else None for i in range(num_agents)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_mod = _get_q_module(agent)
    if q_mod is not None and hasattr(q_mod, "to"):
        try: q_mod.to(device)
        except Exception: pass

    t0 = time.time()
    total_reward = 0.0
    steps = 0
    success_flag = 0
    non_noop_moves = 0
    path_len = 0
    assist_used = 0
    last_assist_step = -10**9

    def _sanitize_action(a, n):
        try: ai = int(a)
        except Exception: ai = 0
        if n > 0: ai = ai % n
        return ai

    for _ in range(max_steps):
        steps += 1
        if time.time() - t0 > timeout_s:
            break

        joint = [noop] * num_agents

        if astar_team is not None:
            for i in range(num_agents):
                if i == target_idx:
                    continue
                try:
                    joint[i] = int(astar_team[i].act(obs[i]))
                except Exception:
                    joint[i] = noop

        try:
            a_sel = agent.select_action(state, eval_mode=True)
            act = int(a_sel) if np.isscalar(a_sel) else int(np.argmax(np.array(a_sel)))
        except Exception:
            act = random.randrange(n_actions)
        joint[target_idx] = act

        # 微辅助（可选）
        if str(assist).lower() == "guidance" and joint[target_idx] == noop:
            allow = True
            if random.random() > float(assist_prob): allow = False
            if allow and (assist_cooldown > 0) and (steps - last_assist_step < assist_cooldown): allow = False
            if allow and (assist_warmup > 0) and (steps <= assist_warmup): allow = False
            max_assists = max(0, int(assist_max_ratio * max_steps))
            if allow and (assist_used >= max_assists): allow = False
            if allow and assist_near_goal > 0 and goal_pos is not None:
                cur_xy, _ = _get_pos_goal_from_env_and_state(env, state, target_idx)
                if cur_xy is not None:
                    mhd = abs(cur_xy[0] - goal_pos[0]) + abs(cur_xy[1] - goal_pos[1])
                    if mhd > int(assist_near_goal): allow = False
            if allow:
                try:
                    gpath = getattr(env, "global_guidance", None)
                    if gpath is not None and len(gpath[target_idx]) > 0:
                        cur_xy, _ = _get_pos_goal_from_env_and_state(env, state, target_idx)
                        if cur_xy is not None:
                            nxt = gpath[target_idx][0]
                            dx, dy = int(nxt[0]-cur_xy[0]), int(nxt[1]-cur_xy[1])
                            joint[target_idx] = toward_with_mapping(dx, dy, mapping)
                            assist_used += 1
                            last_assist_step = steps
                except Exception:
                    pass

        joint = [_sanitize_action(a, n_actions) for a in joint]
        if len(joint) < num_agents: joint += [noop] * (num_agents - len(joint))
        elif len(joint) > num_agents: joint = joint[:num_agents]

        obs_next, reward, terminated, truncated, info = _env_step_joint(env, joint, target_idx)
        try: total_reward += float(reward)
        except Exception: total_reward += float(np.mean(reward))

        if joint[target_idx] != noop:
            non_noop_moves += 1
            path_len += 1

        next_state = obs_next[target_idx] if isinstance(obs_next, (list, tuple)) else obs_next
        obs = obs_next

        pos_now, _ = _get_pos_goal_from_env_and_state(env, next_state, target_idx)
        if pos_now is not None: last_pos = pos_now

        if goal_pos is not None and last_pos is not None and last_pos == goal_pos:
            success_flag = 1
            state = next_state
            break

        if terminated or truncated:
            if isinstance(info, dict):
                info_ai = info.get(target_idx, info) if isinstance(info.get(target_idx, None), dict) else info
                if isinstance(info_ai, dict):
                    for k in ("success", "is_success", "solved", "reached_goal", "done"):
                        if k in info_ai:
                            try:
                                success_flag = int(bool(info_ai[k]))
                                break
                            except Exception:
                                pass
            if success_flag == 0 and goal_pos is not None and last_pos is not None:
                success_flag = int(last_pos == goal_pos)
            state = next_state
            break

        state = next_state

    if opt_len <= 0:
        opt_len = 1
    if success_flag:
        detour_pct = max(0.0, (path_len - opt_len) / opt_len * 100.0)
    else:
        detour_pct = np.nan  # 或者不计入统计


    return {
        "steps": steps,
        "reward": float(total_reward),
        "success": int(success_flag),
        "moving_cost": float(non_noop_moves),
        "detour_pct": float(detour_pct),
        "opt_len": float(opt_len),
    }

# ------------------------- Stage evaluation -------------------------
def evaluate_agent_on_maps(args, stage_id: int, maps: List[MapRecord]) -> pd.DataFrame:
    rows = []
    agent_global = None
    assist_for_stage = 'guidance' if (args.assist == 'guidance' and stage_id <= 1) else 'none'
    assist_prob_for_stage = args.assist_prob if assist_for_stage == 'guidance' else 0.0
    assist_near_goal_for_stage = args.assist_near_goal if assist_for_stage == 'guidance' else 0
    assist_cooldown_for_stage = args.assist_cooldown if assist_for_stage == 'guidance' else 0
    assist_max_ratio_for_stage = args.assist_max_ratio if assist_for_stage == 'guidance' else 0.0
    assist_warmup_for_stage = args.assist_warmup if assist_for_stage == 'guidance' else 0


    for rec in maps:
        cfg = MapConfig(
            size=rec.size,
            num_agents=rec.agents,
            density=rec.density,
            obs_radius=args.obs_radius,
            max_episode_steps=args.max_episode_steps,
            seed=rec.seed,
            max_episode_seconds=args.max_episode_seconds,
        )
        env = _make_env_from_config(cfg)
        _reset_env_with_seed_and_grid(env, rec.seed)

        if agent_global is None:
            # 构建一次 agent（评测用，epsilon=0）
            num_actions = _infer_num_actions(env)
            net = CRNNModel(num_actions=num_actions)
            action_space = _get_action_space(env)
            agent_global = DDQNAgent(net, action_space)
            if torch is not None and os.path.exists(args.model_path):
                try:
                    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=True)
                except TypeError:
                    ckpt = torch.load(args.model_path, map_location="cpu")
                state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
                try:
                    net.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    print("[LoadWarning] net.load_state_dict failed:", e)
            if hasattr(agent_global, "epsilon"):
                agent_global.epsilon = 0.0
        agent = agent_global

        ep_metrics = []
        for _ in range(args.episodes_per_map):
            m = _rollout_one_episode(
                env, agent,
                teammates=args.teammates,
                steps_factor=args.steps_factor,
                secs_factor=args.secs_factor,
                # assist=args.assist,
                # assist_prob=args.assist_prob,
                # assist_cooldown=args.assist_cooldown,
                # assist_max_ratio=args.assist_max_ratio,
                # assist_warmup=args.assist_warmup,
                # assist_near_goal=args.assist_near_goal,
                # >>> 用阶段化后的参数 <<<
                assist=assist_for_stage,
                assist_prob=assist_prob_for_stage,
                assist_cooldown=assist_cooldown_for_stage,
                assist_max_ratio=assist_max_ratio_for_stage,
                assist_warmup=assist_warmup_for_stage,
                assist_near_goal=assist_near_goal_for_stage,
            )
            ep_metrics.append(m)

        success_list = [m['success'] for m in ep_metrics]
        row = {
            'stage': stage_id,
            'map': rec.map_id,
            'size': rec.size,
            'agents': rec.agents,
            'density': rec.density,
            'seed': rec.seed,
            'complexity': rec.complexity,
            'LDD': rec.LDD,
            'BN': rec.BN,
            'MC': rec.MC,
            'DLR': rec.DLR,
            'density_actual': rec.density_actual,
            'episodes': args.episodes_per_map,
            'success_rate': float(np.mean(success_list)) if ep_metrics else 0.0,
            'success_at_k': 1.0 if any(success_list) else 0.0,
            'avg_reward': float(np.mean([m['reward'] for m in ep_metrics])) if ep_metrics else 0.0,
            'avg_steps': float(np.mean([m['steps'] for m in ep_metrics])) if ep_metrics else 0.0,
            'avg_moving_cost': float(np.mean([m['moving_cost'] for m in ep_metrics])) if ep_metrics else 0.0,
            'avg_detour_pct': float(np.mean([m['detour_pct'] for m in ep_metrics])) if ep_metrics else 0.0,
            'avg_opt_len': float(np.mean([m['opt_len'] for m in ep_metrics])) if ep_metrics else 0.0,
        }
        rows.append(row)

    return pd.DataFrame(rows)

# ------------------------- Main -------------------------
def main():
    parser = argparse.ArgumentParser(description="Test trained CL model by map complexity stages (z-score formula)")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--out_csv', type=str, default='test_by_complexity_results.csv')
    parser.add_argument('--num_maps', type=int, default=40)  # 未使用（我们用固定网格组合）；保留接口
    parser.add_argument('--stages', type=int, default=5)
    parser.add_argument('--episodes_per_map', type=int, default=8)

    parser.add_argument('--size_min', type=int, default=16)
    parser.add_argument('--size_max', type=int, default=64)
    parser.add_argument('--agents_min', type=int, default=2)
    parser.add_argument('--agents_max', type=int, default=16)
    parser.add_argument('--density_min', type=float, default=0.05)
    parser.add_argument('--density_max', type=float, default=0.45)

    parser.add_argument('--obs_radius', type=int, default=5)
    parser.add_argument('--max_episode_steps', type=int, default=1200)
    parser.add_argument('--max_episode_seconds', type=int, default=300)

    parser.add_argument('--steps_factor', type=float, default=5.0)
    parser.add_argument('--secs_factor',  type=float, default=1.0)

    parser.add_argument('--teammates', type=str, default='astar', choices=['astar', 'noop'])
    parser.add_argument('--assist', type=str, default='none', choices=['none','guidance'])
    parser.add_argument('--assist_prob', type=float, default=1.0)
    parser.add_argument('--assist_cooldown', type=int, default=0)
    parser.add_argument('--assist_max_ratio', type=float, default=1.0)
    parser.add_argument('--assist_warmup', type=int, default=0)
    parser.add_argument('--assist_near_goal', type=int, default=0)

    parser.add_argument('--max_complexity', type=float, default=None)

    args = parser.parse_args()

    # 1) 采样地图（收集原始特征）
    records = sample_grid_maps()
    if not records:
        raise RuntimeError("No maps sampled.")
    print(f"[INFO] sampled {len(records)} maps")

    # 2) 用样本计算各特征的均值/方差 → 计算 z 复杂度
    stats = compute_feature_stats(records)
    for r in records:
        r.complexity = predict_complexity_z(r, stats)

    # 3) 分阶段（按复杂度升序切分），可选上限过滤
    stage_buckets = split_into_stages(records, args.stages, args.max_complexity)
    if not stage_buckets:
        raise RuntimeError("No valid maps after complexity computation / filtering.")

    # 4) 评测
    all_results: List[pd.DataFrame] = []
    for s, maps in enumerate(stage_buckets):
        if not maps:
            continue
        lo = maps[0].complexity
        hi = maps[-1].complexity
        print(f"▶️ Testing Stage {s} - {len(maps)} maps (complexity {lo:.4f} .. {hi:.4f})")
        df_stage = evaluate_agent_on_maps(args, s, maps)
        all_results.append(df_stage)

    if not all_results:
        print("No results produced.")
        return

    results = pd.concat(all_results, ignore_index=True)

    # 5) 保存明细
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    results.to_csv(args.out_csv, index=False, encoding='utf-8-sig')
    print(f"✅ Saved: {args.out_csv}")

    # 6) 阶段汇总
    summary = results.groupby('stage').agg({
        'success_rate': 'mean',
        'success_at_k': 'mean',
        'avg_reward': 'mean',
        'avg_steps': 'mean',
        'avg_moving_cost': 'mean',
        'avg_detour_pct': 'mean',
        'complexity': ['min', 'max', 'mean']
    })
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    print('\n===== Stage Summary =====')
    print(summary.to_string())

    base, ext = os.path.splitext(args.out_csv)
    summary_path = base + '_stage_summary.csv'
    summary.to_csv(summary_path, encoding='utf-8-sig')
    print(f"✅ Saved: {summary_path}")

if __name__ == '__main__':
    main()

