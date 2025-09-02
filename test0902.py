#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test0902.py — 随机地图评测（强化版）
特性汇总：
1) 严格成功判定：只有真正到达所有目标才记 success
2) 观测归一化：自动把输入张量缩放到 [0,1]
3) 预测让行防抖：在 step 之前预测对向对冲，提前让半数代理 idle
4) 渐进式 A* 解困：目标代理连续卡住即短期交给 A*，有进展逐步归还
5) 易→难动态解锁：先在易图上跑到 SR≥0.6 再自动放大搜索空间
6) 动作顺序锁定：强制按训练时顺序 ['idle','up','down','left','right'] 下发到环境
7) 兼容两种 step 返回格式：(obs,reward,terminated,truncated,info) 或 (obs,reward,dones,info)
8) 参数：--assist-astar, --max-steps-mult, --easy-first, --per-episode-seed

用法示例（PowerShell 单行）：
python .\test0902.py --weights .\models\best_model.pt --episodes 100 --device cpu --out-dir .\eval_runs --per-episode-seed --assist-astar --max-steps-mult 4 --easy-first
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
    AStarAgent = None  # 未安装时，assist/self-rescue 会被跳过

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

    # === 归一化到 [0,1]（避免训练/评测尺度不一致） ===
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
    # 环境显式 success
    if isinstance(info, dict) and (info.get("success") or info.get("finished") or info.get("all_done")):
        return True
    # 全部到达
    try:
        for i in range(len(obs)):
            if tuple(map(int, obs[i]["global_xy"])) != tuple(map(int, obs[i]["global_target_xy"])):
                return False
        return True
    except Exception:
        return False

def build_env(size: int, num_agents: int, density: float, seed: Optional[int], max_steps_mult: int = 4) -> G2RLEnv:
    """只传 __init__ 支持的键，并尽量覆盖内部 max_episode_steps。"""
    ctor = dict(
        size=int(size),
        num_agents=int(num_agents),
        density=float(density),
        seed=(int(seed) if seed is not None else 42),
        obs_radius=7,
        on_target="nothing",
        collission_system="soft",
        max_episode_steps=int(max_steps_mult * int(size)),
    )
    sig = inspect.signature(G2RLEnv.__init__)
    allowed = {p.name for p in sig.parameters.values()
               if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    allowed.discard("self")
    ctor = {k: v for k, v in ctor.items() if k in allowed}
    env = G2RLEnv(**ctor)
    try:
        if hasattr(env, "grid_config"):
            env.grid_config.max_episode_steps = int(max_steps_mult * int(size))
    except Exception:
        pass
    return env

def choose_action_with_compat(agent, obs_i, device_t):
    """兼容多种 act 签名；传不进原始观测就传 tensor；再不行直接贪心。"""
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

# === 对向对冲预测 ===
DELTA_BY_NAME = {
    'idle': (0,0),
    'up':   (-1,0),
    'down': (1,0),
    'left': (0,-1),
    'right':(0,1),
}
def will_back_forth(prev_pos, planned_pos):
    pairs = [(prev_pos[i], planned_pos[i]) for i in range(len(planned_pos))]
    S = set(pairs)
    for u,v in pairs:
        if u != v and (v,u) in S:
            return True
    return False

def apply_delta(pos, action_name):
    dx, dy = DELTA_BY_NAME.get(action_name, (0,0))
    return (pos[0]+dx, pos[1]+dy)

def manhattan(p, q):
    return abs(p[0]-q[0]) + abs(p[1]-q[1])

# ====== 主评测 ======
def evaluate_random(
    weights_path: str,
    episodes: int,
    device: str,
    out_dir: str,
    per_episode_seed: bool,
    assist_astar: bool,
    max_steps_mult: int,
    easy_first: bool,
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

    # 评测必须纯贪心
    if hasattr(agent, "set_epsilon"): agent.set_epsilon(0.0)
    if hasattr(agent, "epsilon"): agent.epsilon = 0.0
    model.eval()
    torch.set_grad_enabled(False)

    # 输出目录
    out_root = ensure_out_dir(out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = out_root / f"eval_{ts}"
    run_path.mkdir(parents=True, exist_ok=True)
    csv_path  = run_path / "episodes.csv"
    json_path = run_path / "summary.json"

    # 写 CSV 头
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "episode","size","num_agents","density",
            "success","steps","collisions_this_run","episode_seed",
            "termination_reason","stuck_steps_target"
        ])

    # 动态易→难
    use_hard = False if easy_first else True
    win = []  # 最近窗口 success
    WIN_SIZE = 50
    UNLOCK_SR = 0.6

    def pick_cfg():
        if use_hard:
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

        # 动作顺序锁定（强制映射为训练顺序）
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

        num_agents = getattr(env, "num_agents", len(obs))
        teammates = []
        target_idx = 0
        for i in range(num_agents):
            if assist_astar and (i != target_idx) and (AStarAgent is not None):
                try:
                    teammates.append(AStarAgent())
                except Exception:
                    teammates.append(None)
            else:
                teammates.append(None)

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

        # 渐进式 A* 接管
        K_STUCK = 3
        ASSIST_WINDOW = 10
        assist_left = 0
        last_goal_dist = None

        with torch.no_grad():
            while True:
                actions = []
                # 策略/队友生成动作
                for i in range(num_agents):
                    if teammates[i] is not None:
                        try:
                            a = int(teammates[i].act(obs[i]))
                        except Exception:
                            a = 0
                        actions.append(a)
                        continue
                    a = choose_action_with_compat(agent, obs[i], device_t)
                    actions.append(int(a))

                # 动作顺序映射到环境索引
                if remap is not None:
                    # actions 里保存的是 EXPECTED 的索引（0..4）
                    actions = [ remap[a] if (0 <= a < len(remap)) else 0 for a in actions ]

                # === 预测对向对冲：在 step 之前让行 ===
                planned = []
                if isinstance(env_actions, list) and len(env_actions) > 0:
                    for i,a in enumerate(actions):
                        name = env_actions[a] if (0 <= a < len(env_actions)) else 'idle'
                        planned.append(apply_delta(prev_pos[i], name))
                    if will_back_forth(prev_pos, planned):
                        for i in range(0, num_agents, 2):
                            actions[i] = 0  # idle

                # === 渐进式 A* 解困：卡住即接管，进展则逐步归还 ===
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

                if (stuck_counter[target_idx] >= K_STUCK) and (AStarAgent is not None):
                    assist_left = ASSIST_WINDOW
                    stuck_counter[target_idx] = 0

                if assist_left > 0 and AStarAgent is not None:
                    try:
                        a_astar = int(AStarAgent().act(obs[target_idx]))
                        actions[target_idx] = a_astar if (remap is None) else (remap[a_astar] if 0 <= a_astar < len(remap) else 0)
                        assist_left -= 1
                        # 若预计能拉近目标距离（用 planned 的目标代理）
                        if isinstance(env_actions, list) and len(env_actions) > 0:
                            name_t = env_actions[actions[target_idx]] if 0 <= actions[target_idx] < len(env_actions) else 'idle'
                            next_pos_t = apply_delta(cur_pos, name_t)
                            if manhattan(next_pos_t, now_goal) < dist_t:
                                assist_left = max(assist_left-2, 0)
                    except Exception:
                        pass
                last_goal_dist = dist_t

                # === step ===
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

                max_steps = getattr(env, "max_episode_steps", None)
                if max_steps is None and hasattr(env, "grid_config") and hasattr(env.grid_config, "max_episode_steps"):
                    max_steps = env.grid_config.max_episode_steps
                if max_steps is None:
                    max_steps = int(max_steps_mult * size)

                # === 严格成功/步数终止 ===
                if reached_goal(obs, info):
                    termination_reason = "success"
                    break
                if step >= max_steps:
                    termination_reason = "max_steps"
                    break

        success = 1 if reached_goal(obs, info) else 0
        sr_list.append(success); steps_list.append(step)

        # 动态易->难解锁
        if easy_first:
            win.append(success)
            if len(win) > WIN_SIZE:
                win.pop(0)
            if (not use_hard) and (len(win) == WIN_SIZE) and (sum(win)/WIN_SIZE >= UNLOCK_SR):
                use_hard = True
                print(f"🔓 已解锁难图：最近{WIN_SIZE}集 SR={sum(win)/WIN_SIZE:.2f}")

        # 写入一行
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                ep, size, nag, f"{den:.6f}",
                success, step, run_collisions, (epi_seed or ""),
                termination_reason, stuck_steps_target
            ])

        print(f"[Episode {ep:04d}] Map(size={size}, agents={nag}, dens={den:.3f}) | "
              f"Success={'✅' if success else '❌'} | Steps={step} | Collisions={run_collisions} | "
              f"Term={termination_reason} | StuckT={stuck_steps_target}")

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
    p.add_argument("--weights", type=str, required=True, help="模型/智能体权重路径（.pt/.pth）")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda", choices=["cpu","cuda"])
    p.add_argument("--out-dir", type=str, default="eval_runs")
    p.add_argument("--per-episode-seed", action="store_true")

    # 新增
    p.add_argument("--assist-astar", action="store_true", help="非目标智能体使用 A* 行为（与常见训练设置对齐）")
    p.add_argument("--max-steps-mult", type=int, default=4, help="max_episode_steps = mult * size（建议 3~6）")
    p.add_argument("--easy-first", action="store_true", help="先在易图采样，达标后自动解锁难图")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_random(
        weights_path=args.weights,
        episodes=args.episodes,
        device=args.device,
        out_dir=args.out_dir,
        per_episode_seed=args.per_episode_seed,
        assist_astar=args.assist_astar,
        max_steps_mult=args.max_steps_mult,
        easy_first=args.easy_first,
    )
