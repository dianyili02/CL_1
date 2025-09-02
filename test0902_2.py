#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test0902.py — 随机地图评测（多模式 + complexity 打印/落盘）

模式开关（互斥默认集）：
  --mode strict    : 纯策略（无掩码/无贪心/无不退步/无对冲/无A*）；max_steps_mult=3
  --mode safemask  : 只开动作掩码
  --mode nonreg    : 掩码 + 朝目标贪心打平 + 不退步
  --mode assisted  : nonreg + 对向让行 + A*接管 + 易→难 + 大图更长步数

复杂度：
  - 若提供 --nn-model，且 NN 的 feature_columns ⊆ {size,num_agents,density,density_actual}，
    则在线计算 nn_pred_success_rate 与 nn_complexity。
  - 否则回退到 proxy_complexity（size/agents/density 的加权启发）。
  - 终端实时打印复杂度，CSV 也会落盘。

示例（系统级评测 + NN复杂度 + 终端打印 + CSV落盘）：
  python .\test0902.py --mode assisted --weights .\models\best_model.pt --episodes 200 --device cpu \
    --out-dir .\eval_runs --per-episode-seed --nn-model "C:\path\to\nn_model.pt"

示例（严格评估模型本体、仅 proxy 复杂度）：
  python .\test0902.py --mode strict --weights .\models\best_model.pt --episodes 200 --device cpu --out-dir .\eval_runs --per-episode-seed
"""

import os, sys, math, random, argparse, json, csv, inspect, time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

# ====== 工程模块 ======
project_root_guess = os.path.abspath(".")
if project_root_guess not in sys.path:
    sys.path.insert(0, project_root_guess)

from g2rl.environment import G2RLEnv
from g2rl.network import CRNNModel
from g2rl.agent import DDQNAgent

try:
    from pogema import AStarAgent
except Exception:
    AStarAgent = None  # 未装时，assist/self-rescue 会跳过

# ====== 采样空间 ======
EASY_SIZES    = [32, 64]
EASY_AGENTS   = [2, 4]
EASY_DENS     = [0.1, 0.3]

FULL_SIZES    = [32, 64, 96, 128]
FULL_AGENTS   = [2, 4, 8, 16]
FULL_DENS     = [0.1, 0.3, 0.5, 0.7]

# ====== Utils ======
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_out_dir(out_dir: str) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _find_array(obj):
    try:
        import torch as _t  # noqa
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
        preferred = ("view_cache","obs","observation","view","state","tensor","grid","image","local_obs","global_obs")
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

def _to_CDHW(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if arr.ndim == 5:
        # [N,C,D,H,W] or [N,D,H,W,C]
        if arr.shape[1] in (1,2,3,4,5,8,11,16):
            arr = arr[0]
        else:
            arr = np.transpose(arr, (0,4,1,2,3))[0]
    elif arr.ndim == 4:
        # [C,D,H,W] or [D,H,W,C]
        if arr.shape[0] in (1,2,3,4,5,8,11,16):
            pass
        elif arr.shape[-1] in (1,2,3,4,5,8,11,16):
            arr = np.transpose(arr, (3,0,1,2))
        else:
            arr = np.transpose(arr, (3,0,1,2))
    elif arr.ndim == 3:
        arr = arr[None, ...]
    elif arr.ndim == 2:
        arr = arr[None, None, ...]
    else:
        raise ValueError(f"无法归一到 [C,D,H,W]：shape={arr.shape}")
    return arr

def obs_to_tensor(s, device, expected_c: int = 11):
    arr = _find_array(s)
    if arr is None:
        raise KeyError("无法从观测中提取输入张量。请确保观测包含可转数组字段，如 view_cache/obs 等。")
    arr = _to_CDHW(arr)  # [C,D,H,W]
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
    x = torch.tensor(arr_fixed[None, ...], dtype=torch.float32, device=device)  # [1,C,D,H,W]
    # === 归一化到 [0,1] ===
    x_min, x_max = float(x.min().item()), float(x.max().item())
    if (x_min < 0.0) or (x_max > 1.0):
        denom = (x_max - x_min) if (x_max - x_min) > 1e-6 else 1.0
        x = (x - x_min) / denom
    return x

def count_collisions(positions_before: List[Tuple[int, int]], positions_after: List[Tuple[int, int]]) -> int:
    c = 0
    uniq = set(positions_after)
    if len(uniq) < len(positions_after):
        c += (len(positions_after) - len(uniq))
    moved = [(positions_before[i], positions_after[i]) for i in range(len(positions_after))]
    for i in range(len(moved)):
        for j in range(i + 1, len(moved)):
            if moved[i][0] == moved[j][1] and moved[i][1] == moved[j][0]:
                c += 1
    return c

def reached_goal(obs, info) -> bool:
    if isinstance(info, dict) and (info.get("success") or info.get("finished") or info.get("all_done")):
        return True
    try:
        for i in range(len(obs)):
            if tuple(map(int, obs[i]["global_xy"])) != tuple(map(int, obs[i]["global_target_xy"])):
                return False
        return True
    except Exception:
        return False

def build_env(size: int, num_agents: int, density: float, seed: Optional[int], max_steps_mult: int = 4) -> G2RLEnv:
    mult = max_steps_mult
    if size >= 96:
        mult = max(mult, 6)  # 大图更长步数
    ctor = dict(
        size=int(size),
        num_agents=int(num_agents),
        density=float(density),
        seed=(int(seed) if seed is not None else 42),
        obs_radius=7,
        on_target="nothing",
        collission_system="soft",
        max_episode_steps=int(mult * int(size)),
    )
    sig = inspect.signature(G2RLEnv.__init__)
    allowed = {p.name for p in sig.parameters.values()
               if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    allowed.discard("self")
    ctor = {k: v for k, v in ctor.items() if k in allowed}
    env = G2RLEnv(**ctor)
    try:
        if hasattr(env, "grid_config"):
            env.grid_config.max_episode_steps = int(mult * int(size))
    except Exception:
        pass
    return env

def choose_action_with_compat(agent, obs_i, device_t):
    if hasattr(agent, "act"):
        try:
            return int(agent.act(obs_i))
        except TypeError:
            try:
                return int(agent.act(obs_i, epsilon=0.0))
            except Exception:
                pass
        except Exception:
            pass
    x = obs_to_tensor(obs_i, device_t, expected_c=11)
    if hasattr(agent, "act"):
        try:
            return int(agent.act(x))
        except TypeError:
            try:
                return int(agent.act(x, epsilon=0.0))
            except Exception:
                pass
        except Exception:
            pass
    with torch.no_grad():
        q = agent.model(x)
        return int(torch.argmax(q, dim=1).item())

# === 位移/距离/对冲检测 ===
DELTA_BY_NAME = {
    'idle': (0,0),
    'up':   (-1,0),
    'down': (1,0),
    'left': (0,-1),
    'right':(0,1),
}
def apply_delta(pos, action_name):
    dx, dy = DELTA_BY_NAME.get(action_name, (0,0))
    return (pos[0]+dx, pos[1]+dy)

def manhattan(p, q):
    return abs(p[0]-q[0]) + abs(p[1]-q[1])

def will_back_forth(prev_pos, planned_pos):
    pairs = [(prev_pos[i], planned_pos[i]) for i in range(len(planned_pos))]
    S = set(pairs)
    for u,v in pairs:
        if u != v and (v,u) in S:
            return True
    return False

# === 掩码/目标启发 ===
def _grid_size_from_obs(obs_i):
    try:
        g = np.array(obs_i.get('global_obstacles', None))
        if g.ndim == 2:
            return int(g.shape[0]), int(g.shape[1])
    except Exception:
        pass
    return None

def _is_occupied(obs_list, pos):
    try:
        for ob in obs_list:
            if tuple(map(int, ob['global_xy'])) == pos:
                return True
    except Exception:
        pass
    return False

def legal_after(obs_i, action_name, HW=None):
    try:
        gobs = np.array(obs_i['global_obstacles']).astype(bool)
        H, W = gobs.shape
    except Exception:
        if HW is None:
            return True
        H, W = HW
        gobs = np.zeros((H, W), dtype=bool)
    x, y = map(int, obs_i['global_xy'])
    dx, dy = DELTA_BY_NAME.get(action_name, (0, 0))
    nx, ny = x + dx, y + dy
    if nx < 0 or ny < 0 or nx >= H or ny >= W:
        return False
    if gobs[nx, ny]:
        return False
    return True

def goal_distance_after(pos, action_name, goal):
    nx, ny = apply_delta(pos, action_name)
    return manhattan((nx, ny), goal)

# ====== NN 复杂度（可选） & Proxy 复杂度 ======
class NNComplexity:
    """
    仅当 feature_columns ⊆ {size,num_agents,density,density_actual} 时启用；
    否则自动禁用，使用 proxy_complexity。
    """
    def __init__(self, model_path: Optional[str], device: str = "cpu"):
        self.enabled = False
        self.device = torch.device(device if (device=="cuda" and torch.cuda.is_available()) else "cpu")
        self.feature_columns = None
        self.means = None
        self.stds  = None
        self.use_sigmoid = True
        if not model_path:
            return
        if not os.path.exists(model_path):
            print(f"⚠️ NN 模型文件不存在：{model_path}，将采用 proxy_complexity。")
            return
        ckpt = torch.load(model_path, map_location="cpu")
        in_dim       = ckpt["in_dim"]
        hidden_dim   = ckpt["hidden_dim"]
        hidden_layers= ckpt["hidden_layers"]
        dropout      = ckpt["dropout"]
        self.use_sigmoid = ckpt.get("use_sigmoid_head", True)
        self.feature_columns = ckpt.get("feature_columns", ckpt.get("features_used"))

        # 仅支持最小特征集合
        min_feats = {"size","num_agents","density","density_actual"}
        if self.feature_columns is None or not set(self.feature_columns).issubset(min_feats):
            print(f"ℹ️ NN 预测跳过：以下特征无法从随机地图即时构造，建议补充： {list(set(self.feature_columns or []) - min_feats)}")
            return

        self.means = np.array(ckpt["scaler_means"], dtype=float)
        self.stds  = np.array(ckpt["scaler_stds"], dtype=float)

        # 构建简单 MLP（不带 backbone 前缀）
        import torch.nn as nn
        layers = []
        dim = in_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            dim = hidden_dim
        layers += [nn.Linear(dim, 1)]
        self.model = nn.Sequential(*layers).to(self.device)
        sd = ckpt["state_dict"]
        # 尝试兼容 "backbone.*" 命名
        mapped = {}
        for k,v in sd.items():
            if k.startswith("backbone."):
                mapped[k[len("backbone."):]] = v
            elif k.startswith("model.backbone."):
                mapped[k[len("model.backbone."):]] = v
            else:
                mapped[k] = v
        self.model.load_state_dict(mapped, strict=False)
        self.model.eval()
        self.enabled = True

    def predict(self, row_dict: dict, clip: bool = True) -> Optional[Tuple[float,float]]:
        if not self.enabled:
            return None
        X = []
        for c in self.feature_columns:
            val = row_dict.get(c, np.nan)
            try:
                val = float(val)
            except Exception:
                val = np.nan
            X.append(val)
        X = np.array(X, dtype=float)
        if np.any(~np.isfinite(X)):
            return None
        stds = np.where(self.stds==0, 1.0, self.stds)
        Xs = (X - self.means) / stds
        with torch.no_grad():
            x = torch.tensor(Xs[None, :], dtype=torch.float32, device=self.device)
            y = self.model(x).squeeze(-1)
            if self.use_sigmoid:
                y = torch.sigmoid(y)
            pred_sr = float(y.item())
            if clip:
                pred_sr = float(np.clip(pred_sr, 0.0, 1.0))
            nn_cpx = float(1.0 - pred_sr)
            if clip:
                nn_cpx = float(np.clip(nn_cpx, 0.0, 1.0))
            return (pred_sr, nn_cpx)

def estimate_density_actual(obs_first) -> Optional[float]:
    try:
        g = np.array(obs_first['global_obstacles']).astype(bool)
        return float(g.mean())
    except Exception:
        return None

def proxy_complexity(size: int, num_agents: int, density_expect: float, density_actual: Optional[float]) -> float:
    """
    轻量启发：0~1（越大越难）
      - size in [32,128]   -> norm_size in [0,1]
      - agents in [2,16]   -> norm_agents in [0,1]
      - density in [0.05,0.75] -> norm_dens in [0,1]
      权重： dens 0.45, agents 0.35, size 0.20
    """
    s = float(size); a = float(num_agents)
    d = float(density_actual) if density_actual is not None else float(density_expect)

    def norm(x, lo, hi):
        if hi == lo: return 0.0
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

    ns = norm(s, 32.0, 128.0)
    na = norm(a, 2.0, 16.0)
    nd = norm(d, 0.05, 0.75)

    c = 0.20*ns + 0.35*na + 0.45*nd
    return float(np.clip(c, 0.0, 1.0))

# ====== 主评测 ======
def evaluate_random(
    weights_path: str,
    episodes: int,
    device: str,
    out_dir: str,
    per_episode_seed: bool,
    # 细粒度开关
    assist_astar: bool,
    mask_actions: bool,
    greedy_tiebreak: bool,
    no_regression: bool,
    anti_swap: bool,
    max_steps_mult: int,
    easy_first: bool,
    # complexity
    nn_model_path: Optional[str],
    nn_clip: bool,
):
    set_seed(42)
    device_t = "cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu"

    # 预热 env & 动作数
    warm_env = build_env(size=32, num_agents=2, density=0.1, seed=123, max_steps_mult=max_steps_mult)
    try:
        action_space = warm_env.get_action_space()
        if hasattr(action_space, "n"):
            n_actions = int(action_space.n)
        elif isinstance(action_space, (list, tuple)):
            n_actions = len(action_space)
        elif isinstance(action_space, int):
            n_actions = action_space
        else:
            n_actions = 5
    except Exception:
        n_actions = 5

    model = CRNNModel(num_actions=n_actions, in_channels=11).to(device_t)

    # 加载 agent / 模型
    agent = None; loaded = False
    if hasattr(DDQNAgent, "load") and callable(getattr(DDQNAgent, "load")):
        try:
            agent = DDQNAgent.load(weights_path, device=device_t)
            agent.model.to(device_t)
            loaded = True
            print(f"✅ 使用 DDQNAgent.load() 加载权重：{weights_path}")
        except Exception as e:
            print(f"⚠️ DDQNAgent.load() 失败，将使用 torch.load: {e}")

    if not loaded:
        sd = torch.load(weights_path, map_location=device_t)
        if isinstance(sd, dict) and "state_dict" in sd:
            model.load_state_dict(sd["state_dict"])
        elif isinstance(sd, dict) and "model_state_dict" in sd:
            model.load_state_dict(sd["model_state_dict"])
        else:
            model.load_state_dict(sd)
        agent = DDQNAgent(model, model, list(range(n_actions)), lr=1e-3, device=device_t)  # 仅用于 act
        agent.model = model
        print(f"✅ 使用 torch.load() 加载模型参数：{weights_path}")

    if hasattr(agent, "set_epsilon"): agent.set_epsilon(0.0)
    if hasattr(agent, "epsilon"): agent.epsilon = 0.0
    model.eval()
    torch.set_grad_enabled(False)

    # NN 复杂度（可选）
    nn_est = NNComplexity(nn_model_path, device=device_t)

    # 输出目录
    out_root = ensure_out_dir(out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = out_root / f"eval_{ts}"
    run_path.mkdir(parents=True, exist_ok=True)
    csv_path  = run_path / "episodes.csv"
    json_path = run_path / "summary.json"

    # 写 CSV 头（新增 complexity 列）
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "episode","size","num_agents","density","density_actual",
            "success","steps","collisions_this_run","episode_seed",
            "termination_reason","stuck_steps_target",
            "nn_pred_success_rate","nn_complexity","proxy_complexity"
        ])

    # 动态易→难
    use_hard = False if easy_first else True
    win = []
    WIN_SIZE = 30
    UNLOCK_SR = 0.7

    def pick_cfg():
        if use_hard:
            if np.random.rand() < 0.3:
                return (int(np.random.choice(EASY_SIZES)),
                        int(np.random.choice(EASY_AGENTS)),
                        float(np.random.choice(EASY_DENS)))
            return (int(np.random.choice(FULL_SIZES)),
                    int(np.random.choice(FULL_AGENTS)),
                    float(np.random.choice(FULL_DENS)))
        else:
            return (int(np.random.choice(EASY_SIZES)),
                    int(np.random.choice(EASY_AGENTS)),
                    float(np.random.choice(EASY_DENS)))

    # 统计
    sr_list = []; steps_list = []
    total_collisions = 0; total_steps = 0

    for ep in range(1, episodes+1):
        if per_episode_seed:
            epi_seed = int(np.random.randint(1_000_000_000))
            np.random.seed(epi_seed); random.seed(epi_seed); torch.manual_seed(epi_seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(epi_seed)
        else:
            epi_seed = None

        size, nag, den = pick_cfg()
        env = build_env(size=size, num_agents=nag, density=den, seed=epi_seed, max_steps_mult=max_steps_mult)

        EXPECTED = ['idle','up','down','left','right']
        remap = None
        env_actions = getattr(env, "actions", None)
        if isinstance(env_actions, list) and env_actions != EXPECTED:
            idx = {name:i for i,name in enumerate(env_actions)}
            if all(n in idx for n in EXPECTED):
                remap = [idx[n] for n in EXPECTED]

        try:
            reset_ret = env.reset()
            if isinstance(reset_ret, tuple) and len(reset_ret) >= 1:
                obs, info = reset_ret[0], (reset_ret[1] if len(reset_ret) > 1 else {})
            else:
                obs, info = reset_ret, {}
        except Exception:
            obs, info = env.reset(), {}

        if ep == 1:
            print("Eval action list:", getattr(env, "actions", None))
            try:
                x0 = obs_to_tensor(obs[0], device_t, expected_c=11)
                print("x0.shape/min/max:", tuple(x0.shape), float(x0.min().item()), float(x0.max().item()))
            except Exception as e:
                print("预处理探针失败：", e)

        # —— 复杂度即时计算（density_actual & NN / proxy）——
        dens_actual = estimate_density_actual(obs[0])
        row_feats = {
            "size": size,
            "num_agents": nag,
            "density": den,
            "density_actual": (dens_actual if dens_actual is not None else den),
        }
        nn_sr_val, nn_cpx_val = (float("nan"), float("nan"))
        if nn_est.enabled:
            res = nn_est.predict(row_feats, clip=nn_clip)
            if res is not None:
                nn_sr_val, nn_cpx_val = res
        proxy_cpx = proxy_complexity(size, nag, den, dens_actual)

        num_agents = getattr(env, "num_agents", len(obs))
        def get_positions(o: List[dict]) -> List[Tuple[int, int]]:
            pos = []
            for i in range(len(o)):
                try:
                    pos.append(tuple(map(int, o[i]["global_xy"])))
                except Exception:
                    pos.append((-10**9, -10**9))
            return pos

        prev_pos = get_positions(obs)
        last_pos = prev_pos[:]
        step = 0
        run_collisions = 0
        stuck_counter = [0]*num_agents
        stuck_steps_target = 0
        termination_reason = "other"

        # A* 接管参数
        K_STUCK = 2
        NO_IMPROVE_K = 5
        ASSIST_WINDOW = 20
        assist_left = 0
        last_goal_dist = None
        no_improve_cnt = 0
        target_idx = 0

        teammates = []
        for i in range(num_agents):
            if (i != target_idx) and AStarAgent is not None and assist_astar:
                try:
                    teammates.append(AStarAgent())
                except Exception:
                    teammates.append(None)
            else:
                teammates.append(None)

        with torch.no_grad():
            while True:
                actions = []
                for i in range(num_agents):
                    if teammates[i] is not None:
                        try:
                            a = int(teammates[i].act(obs[i]))
                        except Exception:
                            a = 0
                        actions.append(a); continue

                    a_raw = choose_action_with_compat(agent, obs[i], device_t)
                    env_actions = getattr(env, "actions", None)
                    action_name = env_actions[a_raw] if isinstance(env_actions, list) and 0 <= a_raw < len(env_actions) else 'idle'

                    chosen = action_name
                    HW = _grid_size_from_obs(obs[i])
                    goal_i = tuple(map(int, obs[i]['global_target_xy']))
                    pos_i  = tuple(map(int, obs[i]['global_xy']))

                    CANDS = ['up','down','left','right','idle']
                    if greedy_tiebreak:
                        CANDS.sort(key=lambda nm: goal_distance_after(pos_i, nm, goal_i))

                    if mask_actions:
                        if (not legal_after(obs[i], chosen, HW)) or _is_occupied(obs, apply_delta(pos_i, chosen)):
                            for nm in CANDS:
                                if legal_after(obs[i], nm, HW) and not _is_occupied(obs, apply_delta(pos_i, nm)):
                                    chosen = nm; break

                    if no_regression:
                        dist_now = manhattan(pos_i, goal_i)
                        dist_next = goal_distance_after(pos_i, chosen, goal_i)
                        if dist_next > dist_now:
                            improved = False
                            for nm in CANDS:
                                if legal_after(obs[i], nm, HW) and not _is_occupied(obs, apply_delta(pos_i, nm)):
                                    if goal_distance_after(pos_i, nm, goal_i) <= dist_now:
                                        chosen = nm; improved = True; break
                            if not improved:
                                chosen = 'idle'

                    if isinstance(env_actions, list) and chosen in env_actions:
                        a = env_actions.index(chosen)
                    else:
                        a = 0
                    actions.append(int(a))

                if remap is not None:
                    actions = [ remap[a] if (0 <= a < len(remap)) else 0 for a in actions ]

                if anti_swap:
                    planned = []
                    env_actions = getattr(env, "actions", None)
                    if isinstance(env_actions, list) and len(env_actions) > 0:
                        for i,a in enumerate(actions):
                            name = env_actions[a] if (0 <= a < len(env_actions)) else 'idle'
                            planned.append(apply_delta(prev_pos[i], name))
                        if will_back_forth(prev_pos, planned):
                            for i in range(0, num_agents, 2):
                                actions[i] = 0  # idle 半数让行

                now_goal = tuple(map(int, obs[target_idx]['global_target_xy']))
                cur_pos  = tuple(map(int, prev_pos[target_idx]))
                dist_t   = manhattan(cur_pos, now_goal)
                if last_goal_dist is None:
                    last_goal_dist = dist_t
                if prev_pos[target_idx] == last_pos[target_idx]:
                    stuck_counter[target_idx] += 1
                else:
                    stuck_counter[target_idx] = 0
                stuck_steps_target += int(prev_pos[target_idx] == last_pos[target_idx])

                no_improve_cnt = (no_improve_cnt + 1) if (dist_t >= last_goal_dist) else 0
                last_goal_dist = min(last_goal_dist, dist_t)

                if assist_astar and (AStarAgent is not None) and (stuck_counter[target_idx] >= K_STUCK or no_improve_cnt >= NO_IMPROVE_K):
                    assist_left = ASSIST_WINDOW
                    stuck_counter[target_idx] = 0
                    no_improve_cnt = 0

                if assist_astar and assist_left > 0 and AStarAgent is not None:
                    try:
                        a_astar = int(AStarAgent().act(obs[target_idx]))
                        actions[target_idx] = a_astar if (remap is None) else (remap[a_astar] if 0 <= a_astar < len(remap) else 0)
                        assist_left -= 1
                    except Exception:
                        pass

                ret = env.step(actions)
                if isinstance(ret, (list, tuple)) and len(ret) == 5:
                    next_obs, rewards, terminated, truncated, info = ret
                    dones_like = np.array(terminated, dtype=bool) | np.array(truncated, dtype=bool)
                    done_all = bool(np.all(dones_like))
                else:
                    next_obs, rewards, dones, info = ret
                    if isinstance(dones, (list, tuple, np.ndarray)):
                        done_all = all(bool(x) for x in dones)
                    elif isinstance(dones, dict) and "all_done" in dones:
                        done_all = bool(dones["all_done"])
                    else:
                        done_all = False

                now_pos = get_positions(next_obs)
                c = count_collisions(prev_pos, now_pos)
                run_collisions += c
                total_collisions += c
                total_steps += 1
                last_pos = prev_pos[:]
                prev_pos = now_pos

                step += 1
                obs = next_obs

                if assist_astar and assist_left > 0 and AStarAgent is not None:
                    new_dist = manhattan(tuple(map(int, prev_pos[target_idx])), now_goal)
                    if new_dist < dist_t:
                        assist_left = max(assist_left - 2, 0)

                max_steps = getattr(env, "max_episode_steps", None)
                if max_steps is None and hasattr(env, "grid_config") and hasattr(env.grid_config, "max_episode_steps"):
                    max_steps = env.grid_config.max_episode_steps
                if max_steps is None:
                    mult = max_steps_mult
                    if size >= 96:
                        mult = max(mult, 6)
                    max_steps = int(mult * size)

                if reached_goal(obs, info):
                    termination_reason = "success"
                    break
                if step >= max_steps:
                    termination_reason = "max_steps"
                    break

        success = 1 if reached_goal(obs, info) else 0
        sr_list.append(success); steps_list.append(step)

        if easy_first:
            win.append(success)
            if len(win) > WIN_SIZE:
                win.pop(0)
            if (not use_hard) and (len(win) == WIN_SIZE) and (sum(win)/WIN_SIZE >= UNLOCK_SR):
                use_hard = True
                print(f"🔓 已解锁难图（混采）：最近{WIN_SIZE}集 SR={sum(win)/WIN_SIZE:.2f}")

        # 落盘（含复杂度）
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                ep, size, nag, f"{den:.6f}",
                ("" if dens_actual is None else f"{dens_actual:.6f}"),
                success, step, run_collisions, (epi_seed or ""),
                termination_reason, stuck_steps_target,
                ("" if not np.isfinite(nn_sr_val) else f"{nn_sr_val:.6f}"),
                ("" if not np.isfinite(nn_cpx_val) else f"{nn_cpx_val:.6f}"),
                f"{proxy_cpx:.6f}"
            ])

        # 终端打印复杂度（优先 NN）
        if np.isfinite(nn_cpx_val):
            cpx_str = f"NN_cpx={nn_cpx_val:.3f}"
        else:
            cpx_str = f"Proxy_cpx={proxy_cpx:.3f}"
        print(f"[Episode {ep:04d}] Map(size={size}, agents={nag}, dens={den:.3f}) | "
              f"Success={'✅' if success else '❌'} | Steps={step} | Collisions={run_collisions} | "
              f"Term={termination_reason} | Cpx={cpx_str}")

    # 汇总
    success_rate = float(np.mean(sr_list)) if len(sr_list) else 0.0
    avg_steps = float(np.mean(steps_list)) if len(steps_list) else math.nan
    collision_rate_per_step = (total_collisions / max(1, total_steps))

    summary = dict(
        episodes=episodes,
        success_rate=round(success_rate, 6),
        avg_steps=round(avg_steps, 3) if not math.isnan(avg_steps) else None,
        collision_per_step=round(collision_rate_per_step, 6),
        device=device_t,
        seed=42,
        per_episode_seed=per_episode_seed,
        assist_astar=assist_astar,
        mask_actions=mask_actions,
        greedy_tiebreak=greedy_tiebreak,
        no_regression=no_regression,
        anti_swap=anti_swap,
        max_steps_mult=max_steps_mult,
        easy_first=easy_first,
        created_at=datetime.now().isoformat(timespec="seconds"),
        weights_path=str(weights_path),
    )

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, ensure_ascii=False, indent=2)

    print("\n===== Evaluation Summary =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n📄 CSV: {csv_path}")
    print(f"🧾 JSON: {json_path}")


def parse_args():
    p = argparse.ArgumentParser()
    # 必需
    p.add_argument("--weights", type=str, required=True, help="模型/智能体权重路径（.pt/.pth）")
    # 通用
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda", choices=["cpu","cuda"])
    p.add_argument("--out-dir", type=str, default="eval_runs")
    p.add_argument("--per-episode-seed", action="store_true")

    # 模式
    p.add_argument("--mode", type=str, default="assisted", choices=["strict","safemask","nonreg","assisted"])

    # 细粒度开关（默认 None，由 mode 决定）
    p.add_argument("--assist-astar", dest="assist_astar", action="store_true")
    p.add_argument("--no-assist-astar", dest="assist_astar", action="store_false")
    p.set_defaults(assist_astar=None)

    p.add_argument("--mask-actions", dest="mask_actions", action="store_true")
    p.add_argument("--no-mask-actions", dest="mask_actions", action="store_false")
    p.set_defaults(mask_actions=None)

    p.add_argument("--greedy-tiebreak", dest="greedy_tiebreak", action="store_true")
    p.add_argument("--no-greedy-tiebreak", dest="greedy_tiebreak", action="store_false")
    p.set_defaults(greedy_tiebreak=None)

    p.add_argument("--no-regression", dest="no_regression", action="store_true")
    p.add_argument("--allow-regression", dest="no_regression", action="store_false")
    p.set_defaults(no_regression=None)

    p.add_argument("--anti-swap", dest="anti_swap", action="store_true")
    p.add_argument("--no-anti-swap", dest="anti_swap", action="store_false")
    p.set_defaults(anti_swap=None)

    p.add_argument("--max-steps-mult", type=int, default=None)
    p.add_argument("--easy-first", dest="easy_first", action="store_true")
    p.add_argument("--no-easy-first", dest="easy_first", action="store_false")
    p.set_defaults(easy_first=None)

    # NN 复杂度相关
    p.add_argument("--nn-model", type=str, default=None, help="可选：NN 模型 checkpoint（用于在线 complexity）")
    p.add_argument("--nn-clip", action="store_true", help="将 NN 预测裁剪到 [0,1]")

    return p.parse_args()


def apply_mode_defaults(args):
    mode = args.mode
    def set_default(name, value):
        if getattr(args, name) is None:
            setattr(args, name, value)

    if mode == "strict":
        set_default("assist_astar", False)
        set_default("mask_actions", False)
        set_default("greedy_tiebreak", False)
        set_default("no_regression", False)
        set_default("anti_swap", False)
        set_default("easy_first", False)
        if args.max_steps_mult is None:
            args.max_steps_mult = 3
    elif mode == "safemask":
        set_default("assist_astar", False)
        set_default("mask_actions", True)
        set_default("greedy_tiebreak", False)
        set_default("no_regression", False)
        set_default("anti_swap", False)
        set_default("easy_first", False)
        if args.max_steps_mult is None:
            args.max_steps_mult = 4
    elif mode == "nonreg":
        set_default("assist_astar", False)
        set_default("mask_actions", True)
        set_default("greedy_tiebreak", True)
        set_default("no_regression", True)
        set_default("anti_swap", False)
        set_default("easy_first", False)
        if args.max_steps_mult is None:
            args.max_steps_mult = 4
    elif mode == "assisted":
        set_default("assist_astar", True)
        set_default("mask_actions", True)
        set_default("greedy_tiebreak", True)
        set_default("no_regression", True)
        set_default("anti_swap", True)
        set_default("easy_first", True)
        if args.max_steps_mult is None:
            args.max_steps_mult = 5
    else:
        if args.max_steps_mult is None:
            args.max_steps_mult = 4
    return args


if __name__ == "__main__":
    args = parse_args()
    args = apply_mode_defaults(args)

    evaluate_random(
        weights_path=args.weights,
        episodes=args.episodes,
        device=args.device,
        out_dir=args.out_dir,
        per_episode_seed=args.per_episode_seed,
        assist_astar=args.assist_astar,
        mask_actions=args.mask_actions,
        greedy_tiebreak=args.greedy_tiebreak,
        no_regression=args.no_regression,
        anti_swap=args.anti_swap,
        max_steps_mult=args.max_steps_mult,
        easy_first=args.easy_first,
        nn_model_path=args.nn_model,
        nn_clip=args.nn_clip,
    )
