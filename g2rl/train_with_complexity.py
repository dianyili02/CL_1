import os
import sys
import csv
import argparse
import yaml
import numpy as np
import torch

# ==== 项目路径修正 ====
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ==== 导入项目内部模块 ====
from g2rl.environment import G2RLEnv
from g2rl.network import CRNNModel
from g2rl.agent import DDQNAgent
from g2rl.complexity import compute_complexity

# ==== 训练函数 ====
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

    for map_info in maps:
        map_name = map_info["name"]
        grid = map_info["grid"]
        num_agents = map_info["num_agents"]

        # === Step 1: 计算 Complexity 并写日志 ===
        comp = compute_complexity(
            grid=grid,
            csv_path=csv_path,
            algo=algo,
            map_name=map_name,
            num_agents=num_agents,
            random_state=random_state
        )
        print(f"[Complexity] {map_name}: {comp['Complexity']:.3f} BN={comp['BN']:.3f} DLR={comp['DLR']:.3f}")

        # === Step 2: 初始化环境和智能体 ===
        env = G2RLEnv(grid=grid, num_agents=num_agents)
        model = CRNNModel(env.observation_space).to(device)
        agent = DDQNAgent(model, env.action_space, device=device)

        # === Step 3: 训练循环 ===
        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            step = 0

            while not done:
                actions = [agent.act(o) for o in obs]
                obs, reward, terminated, truncated, info = env.step(actions)

                total_reward += sum(reward) if isinstance(reward, list) else reward
                step += 1
                done = all(terminated) or all(truncated)

            if (ep + 1) % 10 == 0:
                print(f"[{map_name}] Episode {ep+1}/{num_episodes}, Reward={total_reward:.2f}, Steps={step}")

    print("✅ Training complete.")

# ==== 运行入口 ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="CL", help="Algorithm name (CL or G2RL)")
    parser.add_argument("--maps_yaml", type=str, default="maps.yaml", help="YAML file describing maps")
    parser.add_argument("--csv", type=str, default="train_results.csv", help="Path to CSV log")
    parser.add_argument("--episodes", type=int, default=300, help="Episodes per map")
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    # === 从 YAML 加载地图列表 ===
    with open(args.maps_yaml, "r", encoding="utf-8") as f:
        maps_config = yaml.safe_load(f)

    train(
        algo=args.algo,
        maps=maps_config,
        csv_path=args.csv,
        num_episodes=args.episodes,
        curriculum=args.curriculum,
        device=args.device
    )
