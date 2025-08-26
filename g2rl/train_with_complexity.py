# import os
# import sys
# import csv
# import argparse
# import yaml
# import numpy as np
# import torch

# # ==== 项目路径修正 ====
# project_root = os.path.dirname(os.path.abspath(__file__))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# # ==== 导入项目内部模块 ====
# from g2rl.environment import G2RLEnv
# from g2rl.network import CRNNModel
# from g2rl.agent import DDQNAgent
# from g2rl.complexity import compute_complexity

# # ==== 训练函数 ====
# def train(
#     algo: str,
#     maps: list,
#     csv_path: str,
#     num_episodes: int = 300,
#     curriculum: bool = False,
#     device: str = "cpu",
#     random_state: int = 42
# ):
#     np.random.seed(random_state)
#     torch.manual_seed(random_state)

#     for map_info in maps:
#         map_name = map_info["name"]
#         grid = map_info["grid"]
#         num_agents = map_info["num_agents"]

#         # === Step 1: 计算 Complexity 并写日志 ===
#         comp = compute_complexity(
#             grid=grid,
#             csv_path=csv_path,
#             algo=algo,
#             map_name=map_name,
#             num_agents=num_agents,
#             random_state=random_state
#         )
#         print(f"[Complexity] {map_name}: {comp['Complexity']:.3f} BN={comp['BN']:.3f} DLR={comp['DLR']:.3f}")

#         # === Step 2: 初始化环境和智能体 ===
#         env = G2RLEnv(grid=grid, num_agents=num_agents)
#         model = CRNNModel(env.observation_space).to(device)
#         agent = DDQNAgent(model, env.action_space, device=device)

#         # === Step 3: 训练循环 ===
#         for ep in range(num_episodes):
#             obs, _ = env.reset()
#             done = False
#             total_reward = 0.0
#             step = 0

#             while not done:
#                 actions = [agent.act(o) for o in obs]
#                 obs, reward, terminated, truncated, info = env.step(actions)

#                 total_reward += sum(reward) if isinstance(reward, list) else reward
#                 step += 1
#                 done = all(terminated) or all(truncated)

#             if (ep + 1) % 10 == 0:
#                 print(f"[{map_name}] Episode {ep+1}/{num_episodes}, Reward={total_reward:.2f}, Steps={step}")

#     print("✅ Training complete.")

# # ==== 运行入口 ====
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--algo", type=str, default="CL", help="Algorithm name (CL or G2RL)")
#     parser.add_argument("--maps_yaml", type=str, default="maps.yaml", help="YAML file describing maps")
#     parser.add_argument("--csv", type=str, default="train_results.csv", help="Path to CSV log")
#     parser.add_argument("--episodes", type=int, default=300, help="Episodes per map")
#     parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning")
#     parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
#     args = parser.parse_args()

#     # === 从 YAML 加载地图列表 ===
#     with open(args.maps_yaml, "r", encoding="utf-8") as f:
#         maps_config = yaml.safe_load(f)

#     train(
#         algo=args.algo,
#         maps=maps_config,
#         csv_path=args.csv,
#         num_episodes=args.episodes,
#         curriculum=args.curriculum,
#         device=args.device
#     )

import os
import sys
import csv
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ==== 项目路径修正 ====
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ==== 导入项目内部模块 ====
from g2rl.environment import G2RLEnv
from g2rl.network import CRNNModel
from g2rl.agent import DDQNAgent
from g2rl.complexity import compute_complexity

# ---------------- Utilities ----------------

def _to_tuple_list(x: Optional[List[List[int]]]) -> Optional[List[Tuple[int, int]]]:
    if not x:
        return None
    return [tuple(map(int, p)) for p in x]

def _np_grid(x) -> np.ndarray:
    """确保 grid 是 np.ndarray[uint8]."""
    arr = np.array(x, dtype=np.uint8)
    # 强化：若不是 0/1，做二值化（>0 视为障碍）
    if arr.ndim != 2:
        raise ValueError("grid must be 2D")
    return (arr > 0).astype(np.uint8)

def _append_csv(csv_path: str, row: Dict[str, Any]) -> None:
    if not csv_path:
        return
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(row.keys())
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        w.writerow(row)

# ---------------- Train ----------------

def train(
    algo: str,
    maps: list,
    csv_path: str,
    num_episodes: int = 300,
    curriculum: bool = False,
    device: str = "cpu",
    random_state: int = 42
):
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA 不可用，回落到 CPU")
        device = "cpu"

    # === 先计算每张地图的复杂度（包含 FRA / FDA），并可选排序 ===
    maps_with_scores = []
    for m in maps:
        map_name = m.get("name", "unnamed")
        grid = _np_grid(m["grid"])
        num_agents = int(m.get("num_agents", 2))
        starts_l = m.get("starts", None)  # list 形式保留在 YAML
        goals_l  = m.get("goals", None)

        # 仅传入 compute_complexity 时转 tuple（不改原 YAML 结构）
        starts_t = _to_tuple_list(starts_l)
        goals_t  = _to_tuple_list(goals_l)

        res = compute_complexity(
            grid=grid,
            starts=starts_t,
            goals=goals_t,
            agents=num_agents,
            connectivity=4,
            K=30,
            random_state=random_state,
        )

        # 打印关键指标
        print(f"[Complexity] {map_name}: "
              f"C={res['Complexity']:.3f} | LDD={res['LDD']:.3f} BN={res['BN']:.3f} "
              f"MC={res['MC']:.3f} DLR={res['DLR']:.3f} FRA={res.get('FRA',0.0):.3f} "
              f"FDA={res.get('FDA',0.0):.3f}")

        # 记录，稍后可排序或写 CSV
        maps_with_scores.append((m, res))

        # 可选：落 CSV（每张地图写一行）
        if csv_path:
            csv_row = {
                "algo": algo,
                "map_name": map_name,
                "size": int(m.get("size", max(grid.shape))),
                "num_agents": num_agents,
                "density": float(m.get("density", grid.mean())),
                # 基础 raw
                "Size_raw": float(res.get("Size_raw", max(grid.shape))),
                "Agents_raw": float(res.get("Agents_raw", num_agents)),
                "Density_raw": float(res.get("Density_raw", float(grid.mean()))),
                # 主特征
                "LDD": float(res["LDD"]),
                "BN": float(res["BN"]),
                "MC": float(res["MC"]),
                "DLR": float(res["DLR"]),
                "FRA": float(res.get("FRA", 0.0)),
                "FDA": float(res.get("FDA", 0.0)),
                "FRA_hard": float(res.get("FRA_hard", 0.0)),
                "FDA_ratio": float(res.get("FDA_ratio", 1.0)),
                "Complexity": float(res["Complexity"]),
            }
            _append_csv(csv_path, csv_row)

    # Curriculum：按 Complexity 从低到高排序
    if curriculum:
        maps_with_scores.sort(key=lambda t: t[1]["Complexity"])

    # === 逐图训练 ===
    for m, res in maps_with_scores:
        map_name = m.get("name", "unnamed")
        grid = _np_grid(m["grid"])
        num_agents = int(m.get("num_agents", 2))

        env = G2RLEnv(grid=grid, num_agents=num_agents)
        model = CRNNModel(env.observation_space).to(device)
        agent = DDQNAgent(model, env.action_space, device=device)

        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            step = 0

            while not done:
                # 这里假设 obs 是 list/iterable，每个 agent 一个观测
                actions = [agent.act(o) for o in obs]
                obs, reward, terminated, truncated, info = env.step(actions)

                # 累加回报（reward 可能是 list）
                if isinstance(reward, (list, tuple, np.ndarray)):
                    total_reward += float(np.sum(reward))
                else:
                    total_reward += float(reward)

                step += 1
                # 任务完成或超步
                done = (isinstance(terminated, (list, tuple, np.ndarray)) and all(terminated)) \
                       or (isinstance(truncated, (list, tuple, np.ndarray)) and all(truncated)) \
                       or (terminated is True) or (truncated is True)

            if (ep + 1) % 10 == 0:
                print(f"[{map_name}] Ep {ep+1}/{num_episodes} | Reward={total_reward:.2f} | Steps={step}")

    print("✅ Training complete.")

# ==== 运行入口 ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="CL", help="Algorithm name (CL or G2RL)")
    parser.add_argument("--maps_yaml", type=str, default="maps.yaml", help="YAML file describing maps")
    parser.add_argument("--csv", type=str, default="", help="Path to CSV log (optional)")
    parser.add_argument("--episodes", type=int, default=300, help="Episodes per map")
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning (sort by complexity)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # === 从 YAML 加载地图列表 ===
    with open(args.maps_yaml, "r", encoding="utf-8") as f:
        maps_config = yaml.safe_load(f)

    # maps_config 可以是 dict 或 list，这里统一成 list
    if isinstance(maps_config, dict):
        maps_config = [maps_config]

    train(
        algo=args.algo,
        maps=maps_config,
        csv_path=args.csv,
        num_episodes=args.episodes,
        curriculum=args.curriculum,
        device=args.device,
        random_state=args.seed,
    )
