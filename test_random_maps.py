# -*- coding: utf-8 -*-
import os, sys, argparse, random, inspect, math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict



import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from pogema import AStarAgent

# --- 项目路径 ---
project_root = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from g2rl.environment import G2RLEnv
from g2rl.agent import DDQNAgent
from g2rl.network import CRNNModel
from g2rl import moving_cost, detour_percentage

# --- 你训练时的 complexity 公式（与训练保持一致） ---
from g2rl.complexity_module import compute_map_complexity

INTERCEPT = 0.848
WEIGHTS = {
    'Size': 0.021,
    'Agents': -0.010,
    'Density': 0.077,
    'Density_actual': 0.132,
    'LDD': 0.027,
    'BN': -0.128,
    'MC': 0.039,
    'DLR': 0.002,
}
FEATURE_MEAN_STD = None  # 若训练时做过标准化就在此填入
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_overall_success(out_dir: str, df_all, win: int = 10):
    """
    画“所有地图的总体平均趋势”：
    - x 轴：episode 索引（0..episodes_per_map-1）
    - y 轴：该 episode 索引下跨所有地图的平均成功率
    - 同时画 rolling mean（窗口 win）和 ±1 标准差带
    """
    if "episode" not in df_all.columns or "success" not in df_all.columns:
        print("⚠️ df_all 缺少 episode 或 success 列，跳过 overall 成功率曲线。")
        return

    # 1) 按 episode 索引聚合，得到跨地图的均值/标准差
    g_mean = df_all.groupby("episode")["success"].mean().reset_index(name="mean_sr")
    g_std  = df_all.groupby("episode")["success"].std().reset_index(name="std_sr")
    g = g_mean.merge(g_std, on="episode", how="left").fillna(0.0)

    # 2) 滚动平均（平滑总体趋势）
    roll = g["mean_sr"].rolling(win, min_periods=1).mean()

    # 3) 画图
    plt.figure(figsize=(10, 5))
    plt.plot(g["episode"], g["mean_sr"], label="Mean Success Rate", linewidth=1.5)
    plt.plot(g["episode"], roll, label=f"Rolling Mean (w={win})", linewidth=2)

    # ±1 标准差带（裁剪到 [0,1] 更直观）
    lo = np.clip(g["mean_sr"] - g["std_sr"], 0.0, 1.0)
    hi = np.clip(g["mean_sr"] + g["std_sr"], 0.0, 1.0)
    plt.fill_between(g["episode"], lo, hi, alpha=0.15, label="±1 std")

    plt.xlabel("Episode index (per map)")
    plt.ylabel("Success rate")
    plt.title("Overall Episode Success Across Maps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "overall_success.png")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"📈 Overall 成功率曲线已保存：{out_path}")

# ---------- 随机地图生成 ----------
def sample_random_specs(
    n_maps: int,
    *,
    size_range=(16, 64),
    agents_choices=(2, 4, 8, 16),
    density_range=(0.05, 0.45),
    seed: int = 0
) -> Dict[str, dict]:
    rng = random.Random(seed)
    specs = {}
    for i in range(n_maps):
        size = rng.randint(size_range[0], size_range[1])
        num_agents = rng.choice(agents_choices)
        density = rng.uniform(density_range[0], density_range[1])

        # G2RLEnv: 当 map=None 时会按 size / density 随机生成新网格
        spec = dict(
            map=None,
            size=int(size),
            density=float(density),
            num_agents=int(num_agents),
            obs_radius=7,
            cache_size=4,
            r1=-0.01, r2=-0.1, r3=0.1,
            seed=rng.randint(1, 10**9),
            animation=False,
            collission_system='soft',
            on_target='restart',
            max_episode_steps=64
        )

        # 计算 complexity（仅用基础信息；optional LDD/BN/MC/DLR 没有网格时可不算）
        try:
            cpx, used, raw = compute_map_complexity(
                spec, intercept=INTERCEPT, weights=WEIGHTS,
                feature_mean_std=FEATURE_MEAN_STD, size_mode="max"
            )
            spec["complexity"] = float(cpx)
        except Exception:
            spec["complexity"] = np.nan

        specs[f"rand_{i}"] = spec
    return specs

# ---------- 构造 Env（过滤 __init__ 不认识的键） ----------
def build_env_from_raw(raw_cfg: dict) -> G2RLEnv:
    sig = inspect.signature(G2RLEnv.__init__)
    allowed = {
        p.name for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    allowed.discard("self")

    rename = {}  # 如有命名差异在此映射
    ctor_cfg = {}
    for k, v in raw_cfg.items():
        kk = rename.get(k, k)
        if kk in allowed:
            ctor_cfg[kk] = v
    env = G2RLEnv(**ctor_cfg)

    # 附加不进 __init__ 的字段（这里没有 grid/starts/goals，因为是随机生成）
    return env

# ---------- 载入模型 ----------
def load_model(pt_path, device="cuda"):
    device_t = torch.device("cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu")
    model = CRNNModel().to(device_t)
    ckpt = torch.load(pt_path, map_location=device_t)

    # 兼容字典/整包
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format.")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"load_state_dict: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, device_t



# ---------- 评估一张随机地图 ----------
@torch.no_grad()
# def evaluate_on_map(
#     model: torch.nn.Module,
#     map_name: str,
#     spec: Dict,
#     episodes: int = 20,
#     device: str = "cuda",
#     max_steps: int = 200,
#     max_steps_fn=lambda ep: 50 + 10 * ep
# ) -> Tuple[pd.DataFrame, Dict[str, float]]:
#     model.eval()                # 关掉 Dropout/BN
#     device_t = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
#     env = build_env_from_raw(spec)
#     n_actions = len(env.get_action_space())
#     agent = DDQNAgent(model=model, action_space=list(range(n_actions)), device=device_t)

# # —— 强制无探索，无训练 —— 
#     if hasattr(agent, "epsilon"):      agent.epsilon = 0.0
#     if hasattr(agent, "epsilon_min"):  agent.epsilon_min = 0.0
#     if hasattr(agent, "train_mode"):   agent.train_mode = False   # 如果类里有这个开关
#     if hasattr(agent, "eval_mode"):    agent.eval_mode = True     # 同上
#     # if hasattr(agent, "replay_buffer"): agent.replay_buffer.clear()
#     if hasattr(agent, "retrain"):      
#     # 确保评测循环里绝对不调用 retrain()
#         pass


#     env = build_env_from_raw(spec)
#     action_space = env.get_action_space()
#     agent = DDQNAgent(model=model, action_space=action_space, device=device)

#     logs = []
#     succ_list, steps_list, rew_list, mv_list, dt_list = [], [], [], [], []

#     for ep in tqdm(range(episodes), desc=f"Eval {map_name}", dynamic_ncols=True):
#         obs, info = env.reset()
#         tgt: int = np.random.randint(env.num_agents)
#         target_idx = np.random.randint(env.num_agents)
#         agents = [agent if i == target_idx else AStarAgent() for i in range(env.num_agents)]
#         goal = tuple(env.goals[target_idx])
#         state = obs[target_idx]
#         opt_path = [state['global_xy']] + env.global_guidance[target_idx]

#         success = False
#         total_reward = 0.0
#         T = int(max_steps_fn(ep))
#         mv = float("nan")
#         dt = float("nan")
#         steps = max_steps  # 默认值，如果循环里没 break 就用 max_steps
#         for t in range(max_steps):
#             actions = [a.act(o) for a, o in zip(agents, obs)]
#             obs, reward, terminated, truncated, info = env.step(actions)
#             total_reward += float(reward[tgt])

#             pos = tuple(obs[tgt]['global_xy'])
#             if pos == goal:
#                 success = True
#                 steps = t + 1  # 提前成功，记录实际步数
#                 break
#         else:
#     # 如果没有 break（即失败），steps 保持 max_steps
#             steps = max_steps


#         succ_list.append(1 if success else 0)
#         steps_list.append(t + 1)
#         rew_list.append(total_reward)

#         logs.append({
#             "map": map_name,
#             "agents": env.num_agents,
#             "size": spec.get("size", np.nan),
#             "density": spec.get("density", np.nan),
#             "LDD": spec.get("LDD", np.nan),
#             "BN": spec.get("BN", np.nan),
#             "MC": spec.get("MC", np.nan),
#             "DLR": spec.get("DLR", np.nan),
#             "complexity": spec.get("complexity", np.nan),
#             "episode": ep,
#             "success": int(success),
#             "steps": int(steps),
#             "reward": float(total_reward),
#             "moving_cost": float(mv),
#             "detour_pct": float(dt),
#         })

#     df = pd.DataFrame(logs)
#     summary = {
#         "map": map_name,
#         "size": spec.get("size", np.nan),
#         "density": spec.get("density", np.nan),
#         "LDD": spec.get("LDD", np.nan),
#         "BN": spec.get("BN", np.nan),
#         "MC": spec.get("MC", np.nan),
#         "DLR": spec.get("DLR", np.nan),
#         "agents": spec.get("num_agents", env.num_agents),
#         "complexity": spec.get("complexity", np.nan),
#         "episodes": len(df),
#         "success_rate": float(df["success"].mean()) if len(df) else np.nan,
#         "avg_steps": float(df["steps"].mean()) if len(df) else np.nan,
#         "avg_reward": float(df["reward"].mean()) if len(df) else np.nan,
#         "avg_moving_cost": float(df["moving_cost"].mean()) if len(df) else np.nan,
#         "avg_detour_pct": float(df["detour_pct"].mean()) if len(df) else np.nan,
#     }
#     return df, summary



def evaluate_on_map(
    model: torch.nn.Module,
    map_name: str,
    spec: Dict,
    episodes: int = 20,
    device: str = "cuda",
    max_steps: int = 200,
    max_steps_fn=lambda ep: 50 + 10 * ep
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    在指定地图 spec 上评测若干 episode，返回 (episode-level DataFrame, per-map summary dict)。
    - 评测阶段：无探索、无训练（不调用 store/retrain）。
    - 无论成功与否，都会计算 moving_cost / detour_pct 便于横向比较。
    """
    model.eval()  # 关闭 Dropout/BN
    device_t = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    # 只构造一次环境与智能体
    env = build_env_from_raw(spec)
    n_actions = len(env.get_action_space())
    agent = DDQNAgent(model=model, action_space=list(range(n_actions)), device=device_t)

    # —— 强制无探索，无训练 ——
    if hasattr(agent, "epsilon"):     agent.epsilon = 0.0
    if hasattr(agent, "epsilon_min"): agent.epsilon_min = 0.0
    if hasattr(agent, "train_mode"):  agent.train_mode = False
    if hasattr(agent, "eval_mode"):   agent.eval_mode = True
    # 不要调用 agent.store / agent.retrain

    logs = []

    for ep in tqdm(range(episodes), desc=f"Eval {map_name}", dynamic_ncols=True):
        obs, info = env.reset()

        # 随机选一个被 DDQN 控制的 agent；其余用 A* baseline
        tgt = np.random.randint(env.num_agents)
        goal = tuple(env.goals[tgt])
        state = obs[tgt]

        # 最短路（env 在 reset 时已算好 global_guidance）
        opt_path = [state['global_xy']] + env.global_guidance[tgt]
        opt_len = max(1, len(opt_path) - 1)  # 避免 0

        agents = [agent if i == tgt else AStarAgent() for i in range(env.num_agents)]

        success = False
        total_reward = 0.0

        # 本集步数上限：若提供了 max_steps_fn，用它；否则用固定 max_steps
        this_max_steps = int(max_steps_fn(ep)) if max_steps_fn is not None else int(max_steps)

        steps = this_max_steps  # 先给默认值（失败时会用）
        for t in range(this_max_steps):
            actions = [a.act(o) for a, o in zip(agents, obs)]
            obs, reward, terminated, truncated, info = env.step(actions)
            total_reward += float(reward[tgt])

            # 是否到达
            if tuple(obs[tgt]['global_xy']) == goal:
                success = True
                steps = t + 1  # 记录真实用时
                break

        # 统一计算指标（成功与否都算，便于对比）
        mv = moving_cost(steps, opt_path[0], opt_path[-1])
        dt = detour_percentage(steps, opt_len)

        logs.append({
            "map": map_name,
            "agents": env.num_agents,
            "size": spec.get("size", np.nan),
            "density": spec.get("density", np.nan),
            "LDD": spec.get("LDD", np.nan),
            "BN": spec.get("BN", np.nan),
            "MC": spec.get("MC", np.nan),
            "DLR": spec.get("DLR", np.nan),
            "complexity": spec.get("complexity", np.nan),
            "episode": ep,
            "success": int(success),
            "steps": int(steps),
            "reward": float(total_reward),
            "moving_cost": float(mv),
            "detour_pct": float(dt),
        })

    df = pd.DataFrame(logs)

    summary = {
        "map": map_name,
        "size": spec.get("size", np.nan),
        "density": spec.get("density", np.nan),
        "LDD": spec.get("LDD", np.nan),
        "BN": spec.get("BN", np.nan),
        "MC": spec.get("MC", np.nan),
        "DLR": spec.get("DLR", np.nan),
        "agents": spec.get("num_agents", env.num_agents),
        "complexity": spec.get("complexity", np.nan),
        "episodes": len(df),
        "success_rate": float(df["success"].mean()) if len(df) else np.nan,
        "avg_steps": float(df["steps"].mean()) if len(df) else np.nan,
        "avg_reward": float(df["reward"].mean()) if len(df) else np.nan,
        "avg_moving_cost": float(df["moving_cost"].mean()) if len(df) else np.nan,
        "avg_detour_pct": float(df["detour_pct"].mean()) if len(df) else np.nan,
    }
    return df, summary


# ---------- 可视化 ----------
def _rolling_mean(x, w=50):
    if len(x) == 0:
        return np.array([])
    w = max(1, int(w))
    c = np.cumsum(np.insert(x, 0, 0))
    rm = (c[w:] - c[:-w]) / float(w)
    head = [np.mean(x[:i+1]) for i in range(min(w-1, len(x)))]
    return np.array(head + rm.tolist())

def make_plots(out_dir: str, df_all: pd.DataFrame, df_sum: pd.DataFrame, buckets: int = 5):
    os.makedirs(out_dir, exist_ok=True)

    # 成功率 vs 复杂度（按地图）
    dfp = df_sum.dropna(subset=["complexity"]).sort_values("complexity")
    if len(dfp):
        plt.figure(figsize=(8,5))
        x, y = dfp["complexity"].values, dfp["success_rate"].values
        plt.scatter(x, y, s=40)
        if len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
            coef = np.polyfit(x, y, 1); xx = np.linspace(x.min(), x.max(), 200); yy = np.polyval(coef, xx)
            plt.plot(xx, yy, linewidth=2)
        plt.xlabel("Complexity"); plt.ylabel("Success Rate"); plt.title("Success Rate vs Complexity (per map)")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "success_vs_complexity_scatter.png"), dpi=150); plt.close()

    # 成功率分桶（按复杂度）
    if len(df_sum) and df_sum["complexity"].notna().any():
        dd = df_sum.dropna(subset=["complexity"]).copy()
        edges = np.quantile(dd["complexity"].values, np.linspace(0, 1, buckets+1))
        dd["bucket"] = pd.cut(dd["complexity"], bins=edges, include_lowest=True, right=False)
        sr_by_bucket = dd.groupby("bucket")["success_rate"].mean()
        plt.figure(figsize=(8,5)); sr_by_bucket.plot(kind="bar")
        plt.ylabel("Mean Success Rate"); plt.title(f"Success Rate by Complexity Bucket (maps, buckets={buckets})")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "success_rate_buckets.png"), dpi=150); plt.close()

    # 每集：成功（0/1）与滑窗 SR
    if len(df_all):
        ep = df_all["episode"].values; succ = df_all["success"].values.astype(float)
        plt.figure(figsize=(10,5))
        plt.plot(ep, succ, label="Success (0/1)", linewidth=1)
        rm = _rolling_mean(succ, w=50)
        if len(rm) > 0:
            plt.plot(ep[:len(rm)], rm, linewidth=2, label="Rolling SR (w=50)")
        plt.xlabel("Episode"); plt.ylabel("Success / SR"); plt.title("Episode Success & Rolling SR")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "episode_sr_curve.png"), dpi=150); plt.close()
    
    plot_overall_success(out_dir, df_all, win=10)
import pandas as pd

def summarize_results(episode_logs, out_csv=None, pbar=None):
    """
    根据 episode 日志生成 per-map summary。
    episode_logs: List[dict] or pd.DataFrame
        每条日志至少包含这些字段:
        - map
        - agents
        - complexity
        - steps
        - reward
        - moving_cost (可选)
        - detour_pct (可选)
        - success (0/1)
    """
    df = pd.DataFrame(episode_logs)

    # 每个 map 统计
    summary = df.groupby(["map", "agents", "complexity", "size", "density", "LDD", "BN", "MC", "DLR"]).agg(
        episodes=("success", "count"),
        success_rate=("success", "mean"),
        avg_steps=("steps", "mean"),
        avg_reward=("reward", "mean"),
        avg_moving_cost=("moving_cost", "mean"),
        avg_detour_pct=("detour_pct", "mean"),
    ).reset_index()

    # 排序 (按 complexity)
    summary = summary.sort_values("complexity")

    # 美观打印
    print("\n===== Summary (per-map) =====")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # 可选保存到 CSV
    if out_csv:
        summary.to_csv(out_csv, index=False)
        if pbar:
            pbar.write(f"💾 Saved per-map summary to {out_csv}")

    return summary


def summarize_overall_success(df_all: pd.DataFrame, out_csv: str = None) -> pd.DataFrame:
    """
    输出整体 Success Rate，并附带一些常见切片（按 agents、按 complexity 桶）。
    返回一个 DataFrame（1 行整体 + 多行分组）。
    """
    if df_all.empty:
        print("⚠️ 没有评测数据，无法统计整体成功率。")
        return pd.DataFrame()

    rows = []

    # —— Overall —— #
    overall_sr = df_all["success"].mean()
    rows.append({
        "scope": "OVERALL",
        "key": "ALL",
        "episodes": int(len(df_all)),
        "success_rate": float(overall_sr),
    })

    # —— 按 agents 切片 —— #
    if "agents" in df_all.columns:
        g = df_all.groupby("agents")["success"].mean().reset_index()
        for _, r in g.iterrows():
            rows.append({
                "scope": "BY_AGENTS",
                "key": int(r["agents"]),
                "episodes": int((df_all["agents"] == r["agents"]).sum()),
                "success_rate": float(r["success"]),
            })

    # —— 按 complexity 分桶（5 桶） —— #
    if "complexity" in df_all.columns and df_all["complexity"].notna().any():
        d = df_all.dropna(subset=["complexity"]).copy()
        edges = np.quantile(d["complexity"].values, np.linspace(0, 1, 6))  # 5 桶
        d["cpx_bucket"] = pd.cut(d["complexity"], bins=edges, include_lowest=True, right=False)
        g = d.groupby("cpx_bucket")["success"].mean().reset_index()
        for _, r in g.iterrows():
            rows.append({
                "scope": "BY_COMPLEXITY_BUCKET",
                "key": str(r["cpx_bucket"]),
                "episodes": int((d["cpx_bucket"] == r["cpx_bucket"]).sum()),
                "success_rate": float(r["success"]),
            })

    df_metrics = pd.DataFrame(rows)
    # 美观打印
    print("\n===== Overall Test Success Rate =====")
    print(df_metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if out_csv:
        df_metrics.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"💾 已保存整体/切片成功率：{out_csv}")

    return df_metrics

# ---------- 主入口 ----------
def main():
    ap = argparse.ArgumentParser("Test trained model on RANDOMLY GENERATED maps")
    ap.add_argument("--pt", type=str, required=True, help="模型路径，如 models/model1.pt")
    ap.add_argument("--out_dir", type=str, default="eval_random", help="输出目录")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--n_maps", type=int, default=20, help="随机生成多少张不同地图")
    ap.add_argument("--episodes_per_map", type=int, default=30, help="每张地图评估多少回合")
    ap.add_argument("--size_min", type=int, default=16)
    ap.add_argument("--size_max", type=int, default=64)
    ap.add_argument("--agents", type=int, nargs="+", default=[2,4,8,16])
    ap.add_argument("--dens_min", type=float, default=0.05)
    ap.add_argument("--dens_max", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=200, help="每集最大步数（固定上限）")
    ap.add_argument("--stamp_dir", action="store_true", help="在 out_dir 下再加时间戳子目录，避免覆盖")
    args = ap.parse_args()


    if args.stamp_dir:
        from datetime import datetime
        stamp = datetime.now().strftime('%H-%M-%d-%m-%Y')
        out_dir = os.path.join(args.out_dir, stamp)
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    # os.makedirs(args.out_dir, exist_ok=True)

    # 1) 载入已训练模型
    model, device_t = load_model(args.pt, device=args.device)

    # 2) 随机生成地图规范
    specs = sample_random_specs(
        args.n_maps,
        size_range=(args.size_min, args.size_max),
        agents_choices=tuple(args.agents),
        density_range=(args.dens_min, args.dens_max),
        seed=args.seed
    )

    # 3) 评估
    all_rows = []
    for name, spec in specs.items():
        df_ep, _summ = evaluate_on_map(
            model, name, spec, episodes=args.episodes_per_map, device=str(device_t)
        )
        all_rows.append(df_ep)
        # 逐图保存 episode 级别日志
        df_ep.to_csv(os.path.join(args.out_dir, f"{name}_episodes.csv"), index=False, encoding="utf-8-sig")

    # 合并所有 episode 级别日志
    df_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    all_ep_csv = os.path.join(args.out_dir, "all_episodes.csv")
    df_all.to_csv(all_ep_csv, index=False, encoding="utf-8-sig")

    # 4) 生成 per-map 汇总（你要的 summary 表）
    # 4) 生成 per-map 汇总
    summary_csv = os.path.join(out_dir, "summary.csv")
    df_sum = summarize_results(df_all, out_csv=summary_csv)

# 5) 统计测试环节成功率（整体 + 按 agents + 按 complexity 桶）
    overall_csv = os.path.join(out_dir, "overall_success.csv")
    df_overall = summarize_overall_success(df_all, out_csv=overall_csv)

# 6) 可视化（如果有 make_plots）
    try:
        make_plots(out_dir, df_all, df_sum, buckets=5)
        print(f"\n📊 可视化已保存到：{out_dir}")
    except Exception as e:
        print(f"⚠️ 生成可视化失败：{e}")


    # 5) 可视化
    try:
        make_plots(args.out_dir, df_all, df_sum, buckets=5)
        print(f"\n📊 可视化已保存到：{args.out_dir}")
    except Exception as e:
        print(f"⚠️ 生成可视化失败：{e}")

if __name__ == "__main__":
    main()

