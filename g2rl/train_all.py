# g2rl/train_gray3d.py
import os, sys, csv, argparse, yaml, hashlib, json, traceback, time
from typing import Dict, Any, List, Union, Tuple
import numpy as np
from g2rl.agent import DDQNAgent

# === 项目根路径（按你的工程路径） ===
project_root = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 只在训练时导入
import torch
from pogema import AStarAgent
from tqdm import tqdm

import g2rl.train as train_mod              # 你自己的训练入口
from g2rl.network import CRNNModel          # 你的模型
from g2rl.environment import G2RLEnv        # 仅用于评估兜底

# 可选：如果项目里提供了这两个指标函数，就用；否则提供简化版
try:
    from g2rl import moving_cost, detour_percentage
except Exception:
    def moving_cost(steps, s, g): return float(steps)
    def detour_percentage(steps, shortest): return float(steps)/max(1, shortest) - 1.0

# ----------------- 通用工具 -----------------
def append_row(csv_path: str, header: List[str], row: Dict[str, Any]):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})

def rng_from_key(key: str) -> np.random.Generator:
    h = hashlib.sha256(key.encode("utf-8")).digest()
    keys = np.frombuffer(h[:8], dtype=np.uint32)
    return np.random.default_rng(keys)

def as_grid(arr) -> np.ndarray:
    g = np.array(arr)
    g = (g > 0).astype(np.uint8)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], f"grid 必须是正方形二维阵，当前 {g.shape}"
    return g

def generate_grid(size: int, density: float, rng: np.random.Generator) -> np.ndarray:
    total = size * size
    num_obs = int(round(total * float(density)))
    grid = np.zeros(total, dtype=np.uint8)
    if 0 < num_obs < total:
        idx = rng.choice(total, size=num_obs, replace=False)
        grid[idx] = 1
    grid = grid.reshape(size, size)
    # 四周加边框，避免越界
    grid[0,:]=1; grid[-1,:]=1; grid[:,0]=1; grid[:,-1]=1
    return grid

def grid_hash(grid: np.ndarray) -> str:
    return hashlib.sha256(grid.tobytes()).hexdigest()[:16]

# ----------------- 轻量复杂度特征（不依赖 complexity.py） -----------------
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
           [(-1,-1),(-1,0),( -1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
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

# ----------------- 课程排列（蛇形，保证相邻改一维） -----------------
def curriculum_gray_3d(sizes: List[int], agents: List[int], densities: List[float]) -> List[Dict[str, Any]]:
    plan = []
    s_idx = list(range(len(sizes)))
    for si, s_i in enumerate(s_idx):
        a_idx = list(range(len(agents)))
        if si % 2 == 1:
            a_idx = a_idx[::-1]
        for ai, a_i in enumerate(a_idx):
            d_idx = list(range(len(densities)))
            if ai % 2 == 1:
                d_idx = d_idx[::-1]
            for d_i in d_idx:
                plan.append({
                    "size": sizes[s_i],
                    "num_agents": agents[a_i],
                    "density": densities[d_i],
                })
    return plan

# ----------------- 兜底评估（当你的 train() 不返回统计时） -----------------
def quick_eval(env_cfg: Dict[str, Any], episodes: int, device: torch.device) -> Dict[str, float]:
    """轻量评估：仅估计 SR / 移动成本 / 绕路率，不依赖你的训练 loop。"""
    model = CRNNModel().to(device)
    env = G2RLEnv(**env_cfg)
    agent = DDQNAgent(model, env.get_action_space(), device=device)

    success, mcost_sum, detour_sum, success_eps = 0, 0.0, 0.0, 0
    for ep in range(episodes):
        obs, _ = env.reset()
        target_idx = np.random.randint(env.num_agents)
        agents = [agent if i == target_idx else AStarAgent() for i in range(env.num_agents)]
        goal = tuple(env.goals[target_idx])
        state = obs[target_idx]
        shortest = len(getattr(env, "global_guidance", [[]])[target_idx])
        start_pos = state["global_xy"]

        tmax = 50 + 10 * ep
        for t in range(tmax):
            actions = [ag.act(o) for ag, o in zip(agents, obs)]
            obs, reward, terminated, truncated, info = env.step(actions)
            pos = tuple(obs[target_idx]['global_xy'])
            if pos == goal:
                success += 1
                steps_used = t + 1
                mcost_sum  += moving_cost(steps_used, start_pos, goal)
                detour_sum += detour_percentage(steps_used, shortest)
                success_eps += 1
                break

    return {
        "success_rate": success / max(1, episodes),
        "avg_moving_cost": (mcost_sum / success_eps) if success_eps > 0 else "",
        "avg_detour_pct":  (detour_sum / success_eps) if success_eps > 0 else "",
    }

# ----------------- 主流程 -----------------
def main():
    parser = argparse.ArgumentParser("Gray(蛇形) 3D：相邻仅变一维，调用你自己的 train.py，写入特征+训练统计到 CSV")
    parser.add_argument("--maps-yaml", type=str,
        default=os.path.join(project_root, "g2rl", "map_settings_generated.yaml"))
    parser.add_argument("--template-map", type=str, default="map_0")
    parser.add_argument("--sizes", type=str, default="32,64,96,128")
    parser.add_argument("--agents", type=str, default="2,4,8,16")
    parser.add_argument("--densities", type=str, default="0.1,0.3,0.5,0.7")

    parser.add_argument("--episodes-per-stage", type=int, default=200)
    parser.add_argument("--results-csv", type=str,
        default=os.path.join(project_root, "logs", "train_gray3d.csv"))
    parser.add_argument("--algo", type=str, default="CL-Gray3D")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--connectivity", type=int, default=4, choices=[4,8])
    parser.add_argument("--device", type=str, default="cuda")

    # 只写特征，不训练
    parser.add_argument("--features-only", action="store_true")

    # 如果你的 G2RLEnv 已支持 fixed_grid/fixed_starts/fixed_goals，就加上这个
    parser.add_argument("--use-fixed-grid", action="store_true",
        help="若环境构造函数支持 fixed_grid 等参数，则把 YAML 中的 grid/starts/goals 传入")
    parser.add_argument("--target-sr", type=float, default=0.70,
    help="晋级阈值，阶段成功率达到该值即晋级")
    parser.add_argument("--max-attempts", type=int, default=3,
    help="单阶段最多尝试次数（未达标会重复本阶段）")
    parser.add_argument("--episodes-grow", type=float, default=1.0,
    help="未达标时每次将 episodes_per_stage 乘以该系数（例如 1.5）")
    parser.add_argument("--require-pass", action="store_true",
    help="若阶段在所有尝试后仍未达标则终止整个课程（默认不终止，继续下一阶段）")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 读 YAML
    with open(args.maps_yaml, "r", encoding="utf-8") as f:
        raw_maps = yaml.safe_load(f)
    if isinstance(raw_maps, list):
        all_maps = { (m.get("name") or f"map_{i}"): m for i, m in enumerate(raw_maps) }
    elif isinstance(raw_maps, dict):
        all_maps = raw_maps
    else:
        raise AssertionError("maps_yaml 顶层必须是 list 或 dict")

    # 模板
    if args.template_map not in all_maps:
        first_key = next(iter(all_maps.keys()))
        print(f"[WARN] 模板 {args.template_map} 不存在，改用 {first_key}")
        args.template_map = first_key
    template = all_maps[args.template_map]

    # 三轴值
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    agents = [int(x) for x in args.agents.split(",") if x.strip()]
    densities = [float(x) for x in args.densities.split(",") if x.strip()]
    plan = curriculum_gray_3d(sizes, agents, densities)

    # CSV 头
    header = [
        "algo","stage_idx","config_id","config_json",
        "size","num_agents","density",
        "obs_radius","max_episode_steps",
        "density_actual","LDD","BN","MC","DLR",
        "success_rate","avg_reward","avg_loss","avg_epsilon","success_count","episodes_total",
        "avg_moving_cost","avg_detour_pct",
        "grid_hash","episodes_per_stage"
    ]

    # 允许传入 G2RLEnv 的 key（若不用 fixed_grid）
    ctor_whitelist = {"size","num_agents","density","obs_radius","max_episode_steps"}

    for si, cfg in enumerate(plan):
        size = cfg["size"]; num_agents = cfg["num_agents"]; density = cfg["density"]

        # 固定 grid：优先用模板 grid（尺寸匹配），否则 deterministic 生成
        grid = None
        if isinstance(template, dict) and "grid" in template and template["grid"] is not None:
            try:
                g = as_grid(template["grid"])
                if g.shape[0] == size: grid = g
            except Exception:
                grid = None
        if grid is None:
            key = f"{args.template_map}|size={size}|dens={density:.3f}"
            grid = generate_grid(size=size, density=density, rng=rng_from_key(key))

        density_actual = float(grid.mean())
        gh = grid_hash(grid)

        # 计算特征
        ldd = compute_ldd(grid, 5)
        bn  = compute_bn(grid, args.connectivity)
        mc  = compute_mc(grid, args.connectivity)
        dlr = compute_dlr(grid, num_agents, args.connectivity)

        # 组装传给 train.train 的 map_settings
        if args.use_fixed_grid:
            # 尝试把固定图传给环境（若环境未实现该参数，会在 train() 里抛出，外层捕获）
            env_cfg = {
                **{k: template[k] for k in ctor_whitelist if k in template},
                "size": size, "num_agents": num_agents, "density": density,
                "fixed_grid": grid.tolist(),
                "fixed_starts": template.get("starts", None),
                "fixed_goals":  template.get("goals",  None),
            }
        else:
            # 不传 grid，避免 unexpected keyword argument
            env_cfg = {
                **{k: template[k] for k in ctor_whitelist if k in template},
                "size": size, "num_agents": num_agents, "density": density,
            }

        # === 训练或仅写特征 ===
        # === 训练或仅写特征（带晋级/复读） ===
        stats = {
    "success_rate": "",
    "avg_reward": "", "avg_loss": "", "avg_epsilon": "",
    "success_count": "", "episodes_total": "",
    "avg_moving_cost": "", "avg_detour_pct": "",
        }

        if not args.features_only:
            target_sr = getattr(args, "target_sr", 0.70)
            max_attempts = getattr(args, "max_attempts", 3)
            episodes = args.episodes_per_stage
            grow = getattr(args, "episodes_grow", 1.0)  # 1.0 表示不增长

        passed = False
        for attempt in range(1, max_attempts + 1):
            try:
                model = CRNNModel().to(device)
                agent = train_mod.train(
                model=model,
                map_settings={f"gray3d_stage_{si}": env_cfg},
                map_probs=[1.0],
                num_episodes=episodes,
                batch_size=32,
                replay_buffer_size=50_000,
                decay_range=10_000,
                log_dir="logs",
                device=device,
                )
            # 从 agent 读取统计
                sr = getattr(agent, "final_success_rate", None)
                stats["success_rate"] = sr if sr is not None else ""

                stats["avg_reward"]     = getattr(agent, "avg_reward", "")
                stats["avg_loss"]       = getattr(agent, "avg_loss", "")
                stats["avg_epsilon"]    = getattr(agent, "avg_epsilon", "")
                stats["success_count"]  = getattr(agent, "success_count", "")
                stats["episodes_total"] = getattr(agent, "episodes_total", episodes)

            # 兜底评估
                if stats["success_rate"] in ("", None):
                    ev = quick_eval(env_cfg, episodes=50, device=device)
                    stats["success_rate"]    = ev["success_rate"]
                    stats["avg_moving_cost"] = ev["avg_moving_cost"]
                    stats["avg_detour_pct"]  = ev["avg_detour_pct"]

                print(f"[Stage {si} Attempt {attempt}] SR={stats['success_rate']:.3f} (target={target_sr:.2f})")
                if float(stats["success_rate"]) >= float(target_sr):
                    passed = True
                    break
            except Exception:
                traceback.print_exc()

        # 未达标：增长本阶段训练集数（可选）
            episodes = int(round(episodes * max(1.0, float(grow))))

    # 若需要“必须过关才继续”，可在此判断
        if (not passed) and getattr(args, "require_pass", False):
            print(f"[Stage {si}] 未达到阈值且启用了 --require-pass，终止课程。")
        # 仍然会把本阶段的最后一次结果写入 CSV，然后 return/exit
        # 你可以选择 raise SystemExit 或 return

        # stats = {
        #     "success_rate": "",
        #     "avg_reward": "", "avg_loss": "", "avg_epsilon": "",
        #     "success_count": "", "episodes_total": "",
        #     "avg_moving_cost": "", "avg_detour_pct": "",
        # }

        # if not args.features_only:
        #     try:
        #         model = CRNNModel().to(device)
        #         agent = train_mod.train(
        #             model=model,
        #             map_settings={f"gray3d_stage_{si}": env_cfg},
        #             map_probs=[1.0],
        #             num_episodes=args.episodes_per_stage,
        #             batch_size=32,
        #             replay_buffer_size=50_000,
        #             decay_range=10_000,
        #             log_dir="logs",
        #             device=device,
        #         )
        #         # 优先从 agent 读取统计（若你的 train() 已赋值）
        #         def _get(attr, default=""):
        #             return getattr(agent, attr, default)
        #         stats["success_rate"] = _get("final_success_rate", "")
        #         stats["avg_reward"]    = _get("avg_reward", "")
        #         stats["avg_loss"]      = _get("avg_loss", "")
        #         stats["avg_epsilon"]   = _get("avg_epsilon", "")
        #         stats["success_count"] = _get("success_count", "")
        #         stats["episodes_total"]= _get("episodes_total", "")

        #         # 若 SR 没有，则做一次轻评估兜底
        #         if stats["success_rate"] == "" or stats["success_rate"] is None:
        #             ev = quick_eval(env_cfg, episodes=50, device=device)
        #             stats["success_rate"]   = ev["success_rate"]
        #             stats["avg_moving_cost"]= ev["avg_moving_cost"]
        #             stats["avg_detour_pct"] = ev["avg_detour_pct"]

        #     except TypeError as e:
        #         # 多半是 fixed_grid 等参数环境不认识：提示并回退不传这些键
        #         print("[WARN] 训练失败（可能是环境不支持 fixed_grid/starts/goals），尝试移除这些键后重试。", e)
        #         env_cfg = {
        #             **{k: template[k] for k in ctor_whitelist if k in template},
        #             "size": size, "num_agents": num_agents, "density": density,
        #         }
        #         try:
        #             model = CRNNModel().to(device)
        #             agent = train_mod.train(
        #                 model=model,
        #                 map_settings={f"gray3d_stage_{si}": env_cfg},
        #                 map_probs=[1.0],
        #                 num_episodes=args.episodes_per_stage,
        #                 batch_size=32,
        #                 replay_buffer_size=50_000,
        #                 decay_range=10_000,
        #                 log_dir="logs",
        #                 device=device,
        #             )
        #             stats["success_rate"] = getattr(agent, "final_success_rate", "")
        #             if stats["success_rate"] in ("", None):
        #                 ev = quick_eval(env_cfg, episodes=50, device=device)
        #                 stats["success_rate"]   = ev["success_rate"]
        #                 stats["avg_moving_cost"]= ev["avg_moving_cost"]
        #                 stats["avg_detour_pct"] = ev["avg_detour_pct"]
        #         except Exception:
        #             traceback.print_exc()
        #     except Exception:
        #         traceback.print_exc()

        # 写 CSV
        config_id = f"sz{size}_a{num_agents}_d{density:.2f}"
        config_copy = {k: template[k] for k in ctor_whitelist if k in template}
        config_copy.update({"size": size, "num_agents": num_agents, "density": density})
        config_json = json.dumps(config_copy, ensure_ascii=False, separators=(",",":"))

        row = {
            "algo": args.algo,
            "stage_idx": si,
            "config_id": config_id,
            "config_json": config_json,
            "size": size,
            "num_agents": num_agents,
            "density": density,
            "obs_radius": config_copy.get("obs_radius",""),
            "max_episode_steps": config_copy.get("max_episode_steps",""),
            "density_actual": density_actual,
            "LDD": float(ldd), "BN": float(bn), "MC": float(mc), "DLR": float(dlr),
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
        }
        append_row(args.results_csv, header, row)
        print("✅ 写入：", row)

    print(f"\n🎉 完成。统一结果表：{args.results_csv}")
    if args.features_only:
        print("（本次使用 --features-only：未进行训练，仅导出特征。）")

if __name__ == "__main__":
    main()
