# g2rl/curriculum_fixed.py
from __future__ import annotations
import os, sys, json, csv, yaml, time, hashlib, inspect, argparse
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pogema import AStarAgent

# ==== 工程根路径（按你的实际路径）====
PROJECT_ROOT = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==== 项目模块 ====
from g2rl.environment import G2RLEnv
from g2rl.agent import DDQNAgent
from g2rl.network import CRNNModel

# （可选）项目里若提供这俩函数就引入；没有也不影响训练（仅用于日志）
try:
    from g2rl import moving_cost, detour_percentage
except Exception:
    def moving_cost(steps, s, g):
        return float(steps)
    def detour_percentage(steps, shortest):
        return float(steps) / max(1, shortest) - 1.0

# ----------------- 小工具 -----------------
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def grid_hash(grid: np.ndarray) -> str:
    return hashlib.sha256(grid.astype(np.uint8).tobytes()).hexdigest()[:16]

def as_grid(arr) -> np.ndarray:
    g = (np.array(arr) > 0).astype(np.uint8)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], f"grid 必须是正方形二维阵，got {g.shape}"
    return g

def append_row(csv_path: str, header: List[str], row: Dict[str, Any]):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})

def load_maps_yaml(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        return {(m.get("name") or f"map_{i}"): m for i, m in enumerate(data)}
    if isinstance(data, dict):
        return data
    raise AssertionError("maps_yaml 顶层必须是 list 或 dict")

def sort_curriculum(maps: Dict[str, Dict[str, Any]]) -> List[str]:
    def key_fn(item):
        name, m = item
        size = int(m.get("size", 32))
        agents = int(m.get("num_agents", 2))
        dens = float(m.get("density", 0.1))
        return (size, agents, dens, name)
    return [name for name, _ in sorted(maps.items(), key=key_fn)]

# ----------------- 轻量复杂度特征（0~1 归一化） -----------------
def compute_ldd(grid: np.ndarray, window: int = 5) -> float:
    h, w = grid.shape
    r = window // 2
    if h < window or w < window: return 0.0
    vals = []
    for i in range(r, h-r):
        for j in range(r, w-r):
            vals.append(grid[i-r:i+r+1, j-r:j+r+1].mean())
    if not vals: return 0.0
    v = float(np.var(vals))
    return float(np.clip(v / 0.25, 0.0, 1.0))

def compute_bn(grid: np.ndarray, connectivity: int = 4) -> float:
    free = (grid == 0)
    if free.sum() == 0: return 1.0
    nbrs = [(-1,0),(1,0),(0,-1),(0,1)] if connectivity != 8 else \
           [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    H,W = free.shape
    total=weak=0
    for i in range(H):
        for j in range(W):
            if not free[i,j]: continue
            total += 1
            d = 0
            for di,dj in nbrs:
                ni, nj = i+di, j+dj
                if 0 <= ni < H and 0 <= nj < W and free[ni,nj]:
                    d += 1
            if d <= 2: weak += 1
    return float(weak / max(1,total))

def compute_mc(grid: np.ndarray, connectivity: int = 4) -> float:
    free = (grid == 0)
    if free.sum() == 0: return 0.0
    H,W = free.shape
    vis = np.zeros_like(free, dtype=bool)
    nbrs = [(-1,0),(1,0),(0,-1),(0,1)] if connectivity != 8 else \
           [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    comps = 0
    for i in range(H):
        for j in range(W):
            if free[i,j] and not vis[i,j]:
                comps += 1
                st=[(i,j)]; vis[i,j]=True
                while st:
                    x,y=st.pop()
                    for di,dj in nbrs:
                        nx,ny=x+di,y+dj
                        if 0<=nx<H and 0<=ny<W and free[nx,ny] and not vis[nx,ny]:
                            vis[nx,ny]=True; st.append((nx,ny))
    return float(1.0/comps)

def compute_dlr(grid: np.ndarray, agents: int, connectivity: int = 4) -> float:
    free = int((grid == 0).sum())
    if free <= 0: return 1.0
    density = float(grid.mean())
    risk = (agents**2) / max(1, free) * (1.0 + 3.0 * density)
    return float(np.clip(risk, 0.0, 1.0))

def features_for_csv(grid: np.ndarray, num_agents: int, connectivity: int) -> Dict[str, float]:
    return {
        "LDD": compute_ldd(grid, 5),
        "BN": compute_bn(grid, connectivity),
        "MC": compute_mc(grid, connectivity),
        "DLR": compute_dlr(grid, num_agents, connectivity),
    }

# ----------------- 构造 env 并保证固定 grid -----------------
def build_env_with_fixed(raw_cfg: Dict[str, Any], allow_fixed_starts_goals: bool = True) -> G2RLEnv:
    # 1) 过滤 G2RLEnv.__init__ 支持的参数
    sig = inspect.signature(G2RLEnv.__init__)
    ctor_keys = {p.name for p in sig.parameters.values()
                 if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    ctor_keys.discard("self")
    ctor_cfg = {k: v for k, v in raw_cfg.items() if k in ctor_keys}

    env = G2RLEnv(**ctor_cfg)

    # 2) 固定 YAML 的 grid
    fixed_grid = raw_cfg.get("grid")
    fixed_starts = raw_cfg.get("starts") if allow_fixed_starts_goals else None
    fixed_goals  = raw_cfg.get("goals")  if allow_fixed_starts_goals else None

    if fixed_grid is None:
        return env

    fixed_grid = as_grid(fixed_grid)
    env.fixed_grid = fixed_grid.copy()
    env.size = int(fixed_grid.shape[0])
    env.density = float(fixed_grid.mean())

    # 覆盖可能的随机生成函数
    if hasattr(env, "_generate_grid") and callable(getattr(env, "_generate_grid")):
        def _fixed_gen():
            return env.fixed_grid.copy()
        env._generate_grid = _fixed_gen  # type: ignore

    # 覆盖 reset，确保 reset 后仍为固定图；并可固定起终点
    if hasattr(env, "reset") and callable(getattr(env, "reset")):
        _orig_reset = env.reset
        def _fixed_reset(*args, **kwargs):
            out = _orig_reset(*args, **kwargs)
            env.grid = env.fixed_grid.copy()
            if fixed_starts is not None and fixed_goals is not None:
                env.starts = [tuple(map(int, p)) for p in fixed_starts]
                env.goals  = [tuple(map(int, p)) for p in fixed_goals]
            return out
        env.reset = _fixed_reset  # type: ignore

    return env

# ----------------- 单关训练：返回统计信息字典 -----------------
def train_one_stage(
    model: torch.nn.Module,
    env: G2RLEnv,
    episodes: int,
    batch_size: int,
    decay_range: int,
    replay_buffer_size: int,
    device: str,
    log_dir: str,
    stage_tag: str,
    max_episode_seconds: int = 30,
) -> Dict[str, float]:
    writer = SummaryWriter(log_dir=Path(log_dir) / f"{stage_tag}_{now_tag()}")
    agent = DDQNAgent(
        model=model,
        action_space=env.get_action_space(),
        lr=0.001,
        decay_range=decay_range,
        device=device,
        replay_buffer_size=replay_buffer_size,
    )

    success = 0
    total_reward = 0.0
    total_loss = 0.0
    total_eps = 0.0
    retrain_steps = 0
    sum_mcost_success = 0.0
    sum_detour_success = 0.0
    success_eps_for_cost = 0

    pbar = tqdm(range(episodes), desc=f"Stage {stage_tag}", dynamic_ncols=True)

    for ep in pbar:
        obs, _ = env.reset()
        target_idx = np.random.randint(env.num_agents)
        agents = [agent if i == target_idx else AStarAgent() for i in range(env.num_agents)]

        goal = tuple(env.goals[target_idx])
        state = obs[target_idx]
        shortest = len(getattr(env, "global_guidance", [[]])[target_idx])
        start_pos = state["global_xy"]

        episode_start = time.time()
        timesteps = 50 + 10 * ep
        scalars = {"Reward": 0.0, "Average Loss": 0.0, "Average Epsilon": 0.0, "Success": 0}

        for t in range(timesteps):
            if time.time() - episode_start > max_episode_seconds:
                break

            actions = [ag.act(o) for ag, o in zip(agents, obs)]
            obs, reward, terminated, truncated, info = env.step(actions)

            pos = tuple(obs[target_idx]["global_xy"])
            done = (pos == goal)
            terminated[target_idx] = done

            agent.store(state, actions[target_idx], reward[target_idx], obs[target_idx], terminated[target_idx])
            state = obs[target_idx]
            scalars["Reward"] += float(reward[target_idx])

            if len(agent.replay_buffer) >= batch_size:
                retrain_steps += 1
                loss_val = float(agent.retrain(batch_size))
                total_loss += loss_val
                total_eps += float(agent.epsilon)

            if done:
                scalars["Success"] = 1
                success += 1
                # 成功才统计这两个
                steps_used = t + 1
                sum_mcost_success += moving_cost(steps_used, start_pos, goal)
                sum_detour_success += detour_percentage(steps_used, shortest)
                success_eps_for_cost += 1
                break

        total_reward += scalars["Reward"]

        # 逐 episode 的标量也写入 TensorBoard
        writer.add_scalar("Reward", scalars["Reward"], ep)
        writer.add_scalar("Success", scalars["Success"], ep)
        if retrain_steps > 0:
            writer.add_scalar("AvgLoss_running", total_loss / retrain_steps, ep)
            writer.add_scalar("AvgEpsilon_running", total_eps / retrain_steps, ep)

        pbar.set_postfix(SR=f"{success / (ep + 1):.2f}",
                         R=f"{scalars['Reward']:.2f}")

    avg_reward = total_reward / max(1, episodes)
    avg_loss = total_loss / max(1, retrain_steps)
    avg_epsilon = total_eps / max(1, retrain_steps)
    avg_mcost = (sum_mcost_success / success_eps_for_cost) if success_eps_for_cost > 0 else ""
    avg_detour = (sum_detour_success / success_eps_for_cost) if success_eps_for_cost > 0 else ""

    writer.close()
    return {
        "success_rate": success / max(1, episodes),
        "avg_reward": avg_reward,
        "avg_loss": avg_loss,
        "avg_epsilon": avg_epsilon,
        "success_count": success,
        "episodes_total": episodes,
        "avg_moving_cost": avg_mcost,
        "avg_detour_pct": avg_detour,
    }

# ----------------- 主流程：YAML → Curriculum → 多尝试晋级 + CSV -----------------
def main():
    ap = argparse.ArgumentParser("Curriculum on fixed YAML grids (features + SR + training stats to CSV)")
    ap.add_argument("--maps-yaml", type=str,
                    default=os.path.join(PROJECT_ROOT, "g2rl", "map_settings_generated.yaml"))
    ap.add_argument("--out-csv", type=str,
                    default=os.path.join(PROJECT_ROOT, "logs", "curriculum_features_sr.csv"))
    ap.add_argument("--episodes-per-stage", type=int, default=200)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--target-sr", type=float, default=0.70)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--replay-buffer-size", type=int, default=50_000)
    ap.add_argument("--decay-range", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log-dir", type=str, default="logs")
    ap.add_argument("--connectivity", type=int, default=4, choices=[4,8])
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    maps = load_maps_yaml(args.maps_yaml)
    plan = sort_curriculum(maps)

    header = [
        "algo","stage_idx","attempt","map_name","config_id","config_json",
        "size","num_agents","density","obs_radius","max_episode_steps",
        "density_actual","LDD","BN","MC","DLR",
        "success_rate","avg_reward","avg_loss","avg_epsilon","success_count","episodes_total",
        "avg_moving_cost","avg_detour_pct",
        "grid_hash","episodes_per_stage","promoted"
    ]

    model = CRNNModel().to(device)
    algo_name = "CL-FixedGrid"
    stage_idx = 0

    for name in plan:
        raw = maps[name]
        env = build_env_with_fixed(raw, allow_fixed_starts_goals=True)

        size = int(raw.get("size", env.size))
        num_agents = int(raw.get("num_agents", 2))
        density = float(raw.get("density", env.density))
        obs_radius = int(raw.get("obs_radius", 5))
        max_steps = int(raw.get("max_episode_steps", 100))

        try:
            grid = as_grid(raw["grid"])
        except Exception:
            grid = getattr(env, "fixed_grid", getattr(env, "grid", None))
            if grid is None:
                raise RuntimeError(f"{name} 无法获得 grid 以写日志。请保证 YAML 含 grid。")
        density_actual = float(grid.mean())
        gh = grid_hash(grid)

        feats = features_for_csv(grid, num_agents, args.connectivity)

        promoted = False
        for attempt in range(1, args.max_attempts + 1):
            stats = train_one_stage(
                model=model,
                env=env,
                episodes=args.episodes_per_stage,
                batch_size=args.batch_size,
                decay_range=args.decay_range,
                replay_buffer_size=args.replay_buffer_size,
                device=str(device),
                log_dir=args.log_dir,
                stage_tag=f"{stage_idx}-{name}-try{attempt}",
            )

            config_id = f"sz{size}_a{num_agents}_d{density:.2f}"
            config_json = json.dumps({
                "size": size, "num_agents": num_agents, "density": density,
                "obs_radius": obs_radius, "max_episode_steps": max_steps
            }, ensure_ascii=False)

            row = {
                "algo": algo_name,
                "stage_idx": stage_idx,
                "attempt": attempt,
                "map_name": name,
                "config_id": config_id,
                "config_json": config_json,
                "size": size,
                "num_agents": num_agents,
                "density": density,
                "obs_radius": obs_radius,
                "max_episode_steps": max_steps,
                "density_actual": density_actual,
                "LDD": feats["LDD"], "BN": feats["BN"], "MC": feats["MC"], "DLR": feats["DLR"],
                "success_rate": stats["success_rate"],
                "avg_reward": stats["avg_reward"],
                "avg_loss": stats["avg_loss"],
                "avg_epsilon": stats["avg_epsilon"],
                "success_count": stats["success_count"],
                "episodes_total": stats["episodes_total"],
                "avg_moving_cost": stats["avg_moving_cost"],
                "avg_detour_pct": stats["avg_detour_pct"],
                "grid_hash": gh,
                "episodes_per_stage": args.episodes_per_stage,
                "promoted": "" if stats["success_rate"] < args.target_sr else "yes",
            }
            append_row(args.out_csv, header, row)
            print("✅ 写入：", row)

            if stats["success_rate"] >= args.target_sr:
                promoted = True
                break

        if not promoted:
            print(f"[WARN] 阶段 {stage_idx}（{name}）未达阈值 {args.target_sr:.2f}，继续下一关。")
        else:
            print(f"[OK] 阶段 {stage_idx}（{name}）晋级：SR={stats['success_rate']:.3f} ≥ {args.target_sr:.2f}")

        stage_idx += 1

    print(f"\n🎉 完成。结果表：{args.out_csv}")
    os.makedirs(os.path.join(PROJECT_ROOT, "ckpts"), exist_ok=True)
    torch.save({"state_dict": model.state_dict()},
               os.path.join(PROJECT_ROOT, "ckpts", f"{algo_name}_{now_tag()}.pt"))

if __name__ == "__main__":
    # 双击运行：无参时自动带默认参数
    if len(sys.argv) == 1:
        sys.argv.extend([
            "--maps-yaml", os.path.join(PROJECT_ROOT, "g2rl", "map_settings_generated.yaml"),
            "--out-csv",   os.path.join(PROJECT_ROOT, "logs", "curriculum_features_sr.csv"),
            "--episodes-per-stage", "200",
            "--max-attempts", "3",
            "--target-sr", "0.70",
            "--device", "cuda",
            "--seed", "42",
        ])
    # 自动切到虚拟环境（可选 .venv38）
    venv_python = os.path.abspath(os.path.join(PROJECT_ROOT, ".venv38", "Scripts", "python.exe"))
    if os.path.exists(venv_python) and os.path.abspath(sys.executable) != venv_python:
        os.execv(venv_python, [venv_python] + sys.argv)
    main()
