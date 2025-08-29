from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional, Union
import numpy as np
import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from pogema import AStarAgent
project_root = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main - train"
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from g2rl.environment import G2RLEnv
from g2rl.agent import DDQNAgent
from g2rl.network import CRNNModel
from g2rl import moving_cost, detour_percentage
from g2rl.curriculum import CurriculumScheduler
import time
from pathlib import Path
from collections import deque
import pandas as pd

MAP_SETTINGS_PATH = 'C:/Users/MSc_SEIoT_1/MAPF_G2RL-main - train/g2rl/map_settings_generated_new.yaml'

import yaml


from g2rl.map_settings import predict_complexity  

# 新增
import argparse
import csv
# 顶部补充：
import inspect
import numpy as np
import yaml

# ...（你的其他 import 保持不变）

# 读取 YAML（保持你原来的路径变量）
# 读取 YAML（保持你原有的 MAP_SETTINGS_PATH）
with open(MAP_SETTINGS_PATH, "r", encoding="utf-8") as f:
    base_map_settings = yaml.safe_load(f)

# 兼容顶层为 list 的情况：用 name 做 key 转成 dict
if isinstance(base_map_settings, list):
    base_map_settings = {
        (m.get("name") or f"map_{i}"): m
        for i, m in enumerate(base_map_settings)
    }

# 如果你已有 map_names 的筛选逻辑，保留你的；否则用全部
map_names = list(base_map_settings.keys())

# ====== 关键修正：过滤掉 G2RLEnv 不认识的参数 ======
import inspect
import numpy as np

# 取 G2RLEnv.__init__ 支持的参数名（或你也可以手写白名单）
_sig = inspect.signature(G2RLEnv.__init__)
_ctor_params = {
    p.name for p in _sig.parameters.values()
    if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
}
_ctor_params.discard("self")

# 若 YAML 键名与构造函数不一致，可在这里做映射
# 例如：YAML 用 'size'，构造函数叫 'map_size'，就写 {'size':'map_size'}
_rename = {
    # 'size': 'map_size',
    # 'num_agents': 'n_agents',
}

def _build_ctor_cfg(raw: dict) -> dict:
    tmp = { _rename.get(k, k): v for k, v in raw.items() }
    return { k: v for k, v in tmp.items() if k in _ctor_params }

maps = []
for name in map_names:
    raw = base_map_settings[name]

    # 1) 只传 __init__ 认识的键
    ctor_cfg = _build_ctor_cfg(raw)
    env = G2RLEnv(**ctor_cfg)

    # 2) 将 grid/starts/goals 等“额外信息”作为属性挂在 env 上（不会传给 __init__）
    if "grid" in raw:
        try:
            env.grid = (np.array(raw["grid"]) > 0).astype(np.uint8)
        except Exception:
            env.grid = None
    if "starts" in raw:
        env.starts = raw["starts"]  # list[list[int,int]] 或 None
    if "goals" in raw:
        env.goals = raw["goals"]

    maps.append(env)
# ====== 修正结束 ======

# ===== Complexity-based Curriculum Scheduler =====
from typing import Iterable
from g2rl.complexity import compute_complexity
import numpy as np
import yaml, math, random

# --- 你的线性公式 ---
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
FEATURE_MEAN_STD = None  # 若训练时做过标准化，按 {'Size':(mu, sigma), ...} 传入

# def _compute_complexities_for_settings(base_map_settings: Dict[str, dict],
#                                        size_mode: str = "max") -> pd.DataFrame:
#     """对单 YAML 内所有地图项逐个计算 complexity，返回 DataFrame."""
#     rows = []
#     for name, spec in base_map_settings.items():
#         try:
#             cpx, used, raw = compute_complexity(
#                 spec, intercept=INTERCEPT, weights=WEIGHTS,
#                 feature_mean_std=FEATURE_MEAN_STD, size_mode=size_mode
#             )
#             rows.append({
#                 "name": name,
#                 "complexity": float(cpx),
#                 "spec": spec,
#             })
#         except Exception as e:
#             rows.append({"name": name, "error": str(e), "spec": spec})
#     df = pd.DataFrame(rows)
#     if "error" in df.columns:
#         df = df[df["error"].isna()]
#     return df.sort_values("complexity").reset_index(drop=True)

def _build_stages_by_quantile_df(df: pd.DataFrame, n_stages: int = 5, min_per_stage: int = 5):
    if len(df) == 0:
        raise ValueError("没有可用地图用于 complexity 课程。")
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
            "items": sub.to_dict("records"),  # 每条含 name/spec/complexity
        })
    return stages


# ======== Robust complexity building (single source of truth) ========
import math, random, numpy as np, pandas as pd
from typing import Dict, Tuple

# 统一键名映射（YAML里常见大小写/别名）
_KEY_RENAME = {
    "map_size": "size",
    "Size": "size",
    "Agents": "num_agents",
    "n_agents": "num_agents",
    "N_agents": "num_agents",
    "Density": "density",
    "obs_density": "density",
}

def _normalize_spec(raw: dict) -> Tuple[dict, list]:
    """把一条 map spec 统一成 {size, num_agents, density, ...} 并返回(规范化spec, 缺失字段列表)"""
    spec = {}
    missing = []
    # 先 rename
    for k, v in raw.items():
        spec[_KEY_RENAME.get(k, k)] = v
    # 必要字段
    for k in ("size", "num_agents", "density"):
        if k not in spec:
            missing.append(k)
    return spec, missing

def _fallback_complexity(spec: dict) -> float:
    """compute_complexity 失败时的后备简式，至少保证有数"""
    size = int(spec.get("size", 8))
    na   = int(spec.get("num_agents", 2))
    dens = float(spec.get("density", 0.10))
    # 你可以改这公式；先用一个温和上升的基线
    return 0.45 * (na / max(1, size)) + 0.45 * dens + 0.10 * math.log2(max(2, size))

def _compute_complexities_for_settings(base_map_settings: Dict[str, dict],
                                       size_mode: str = "max") -> pd.DataFrame:
    rows, err_rows = [], []
    for name, raw in base_map_settings.items():
        spec, missing = _normalize_spec(raw if isinstance(raw, dict) else {})
        if missing:
            err_rows.append((name, f"missing keys: {missing}"))
            # 仍然用 fallback 计算，避免整表无 complexity
            cpx = _fallback_complexity(spec)
            rows.append({"name": name, "complexity": float(cpx), "spec": spec, "note": "fallback_missing"})
            continue

        # 优先尝试你的 learn-based 复杂度
        try:
            # 注意：compute_complexity 的 spec 需要是“你实现所需的键”，
            # 这里传规范化后的 spec；如它还需要其他字段，请在 _normalize_spec 里补齐/rename
            cpx, used, raw_feat = compute_complexity(
                spec, intercept=INTERCEPT, weights=WEIGHTS,
                feature_mean_std=FEATURE_MEAN_STD, size_mode=size_mode
            )
            rows.append({"name": name, "complexity": float(cpx), "spec": spec})
        except Exception as e:
            # 回退到简式，记录错误原因
            err_rows.append((name, str(e)))
            cpx = _fallback_complexity(spec)
            rows.append({"name": name, "complexity": float(cpx), "spec": spec, "note": "fallback_error"})

    df = pd.DataFrame(rows)

    # 最终保险：如果依然没有 'complexity' 列，直接报带预览的错
    if "complexity" not in df.columns:
        raise RuntimeError(f"No 'complexity' column produced. Preview:\n{df.head()}")

    # 清洗非法值
    df = df.dropna(subset=["complexity"])
    df = df[np.isfinite(df["complexity"])].copy()
    df = df.sort_values("complexity").reset_index(drop=True)

    # 打印错误统计，帮助你排查为何使用了 fallback
    if err_rows:
        print("⚠️  compute_complexity() errors (fallback used):")
        for nm, msg in err_rows[:10]:
            print(f"  - {nm}: {msg}")
        if len(err_rows) > 10:
            print(f"  ... and {len(err_rows)-10} more.")

    print(f"✅ Complexity DF ready: {len(df)} maps, columns={list(df.columns)}")
    return df
# ======== end robust builder ========


class ComplexityScheduler:
    """
    与你 train() 中使用的 scheduler 接口兼容：
      - properties: current_stage, max_stage, episodes_per_stage, threshold
      - methods: get_updated_map_settings(), add_episode_result(x), ready_to_advance(),
                 advance(pbar), repeat_stage(pbar), is_done(), current_window_sr()
    """
    def __init__(self,
                 base_map_settings: Dict[str, dict],
                 n_stages: int = 5,
                 min_per_stage: int = 5,
                 episodes_per_stage: int = 100,
                 threshold: float = 0.70,
                 window_size: int = 100,
                 shuffle_each_stage: bool = True,
                 seed: int = 0,
                 size_mode: str = "max"):
        self.episodes_per_stage = episodes_per_stage
        self.threshold = threshold
        self.window_size = window_size

        self._rng = random.Random(seed)
        df = _compute_complexities_for_settings(base_map_settings, size_mode=size_mode)
        stages = _build_stages_by_quantile_df(df, n_stages=n_stages, min_per_stage=min_per_stage)

        # build iterable per stage
        self._stage_items = []
        for st in stages:
            items = list(st["items"])
            if shuffle_each_stage:
                self._rng.shuffle(items)
            self._stage_items.append(items)

        self._stage_edges = [(st["cpx_min"], st["cpx_max"]) for st in stages]
        self.current_stage = 0
        self.max_stage = len(self._stage_items) - 1

        self._idx_in_stage = 0
        self._win = deque(maxlen=self.window_size)
        self._stage_ep_cnt = 0
        self._stage_succ_cnt = 0

    def get_updated_map_settings(self) -> Dict[str, dict]:
        """返回 {map_name: spec}（你 train() 里原样接收）"""
        if self.current_stage > self.max_stage:
            return {}
        items = self._stage_items[self.current_stage]
        if not items:
            raise RuntimeError(f"Stage {self.current_stage} 没有地图。")
        item = items[self._idx_in_stage]
        self._idx_in_stage = (self._idx_in_stage + 1) % len(items)
        return {item["name"]: item["spec"]}

    def add_episode_result(self, success: int):
        self._win.append(1 if success else 0)
        self._stage_ep_cnt += 1
        if success:
            self._stage_succ_cnt += 1

    def current_window_sr(self) -> float:
        if len(self._win) == 0:
            return 0.0
        return float(sum(self._win) / len(self._win))

    def ready_to_advance(self) -> bool:
        # 可选：用滑窗直接判定
        return self.current_window_sr() >= self.threshold and self._stage_ep_cnt >= self.episodes_per_stage//2

    def advance(self, pbar=None):
        if pbar:
            pbar.write(f"✅ 进入下一阶段：{self.current_stage} -> {self.current_stage + 1}")
        self.current_stage += 1
        self._idx_in_stage = 0
        self._win.clear()
        self._stage_ep_cnt = 0
        self._stage_succ_cnt = 0

    def repeat_stage(self, pbar=None):
        if pbar:
            pbar.write(f"🔁 阶段 {self.current_stage} 未达标，重复训练该阶段")
        # 重置计数，但不变更阶段
        self._idx_in_stage = 0
        self._win.clear()
        self._stage_ep_cnt = 0
        self._stage_succ_cnt = 0

    def is_done(self) -> bool:
        return self.current_stage > self.max_stage

# 后续逻辑保持不变（把 maps 传给你的训练/调度流程）

def append_result_row(csv_path, row_dict, header_order):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header_order)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row_dict.get(k, "") for k in header_order})


import pandas as pd
os.makedirs("plots", exist_ok=True)

# 设置正确路径
correct_path = "C:/Users/MSc_SEIoT_1/MAPF_G2RL-main"
if correct_path not in sys.path:
    sys.path.insert(0, correct_path)

# 确认导入的是你修改的文件

# with open("g2rl/map_settings_generated.yaml", "r", encoding="utf-8") as f:
#     base_map_settings = yaml.safe_load(f)
# if isinstance(base_map_settings, list):
#     base_map_settings = {
#         (m.get("name") or f"map_{i}"): m
#         for i, m in enumerate(base_map_settings)
#     }
# scheduler = CurriculumScheduler(
#     base_map_settings=base_map_settings,
#     agent_range=[4, 6, 8, 16],  # 每个阶段的 agent 数量
#     episodes_per_stage=100,     # 每 100 episodes 提升一阶段
# )

# 读 YAML（你已有）
with open(MAP_SETTINGS_PATH, "r", encoding="utf-8") as f:
    base_map_settings = yaml.safe_load(f)

# 顶层是 list 的情况转成 {name: spec}
if isinstance(base_map_settings, list):
    base_map_settings = { (m.get("name") or f"map_{i}"): m for i, m in enumerate(base_map_settings) }

# 用 complexity 划分课程
scheduler = ComplexityScheduler(
    base_map_settings=base_map_settings,
    n_stages=5,
    min_per_stage=10,
    episodes_per_stage=100,
    threshold=0.70,
    window_size=100,
    shuffle_each_stage=True,
    seed=0,
    size_mode="max",  # 注意与训练时的 Size 定义一致
)







def evaluate_model_inline(model, env_config, num_episodes=20, device='cuda'):
    from pogema import AStarAgent
    from g2rl.environment import G2RLEnv
    from g2rl.agent import DDQNAgent
    from g2rl import moving_cost, detour_percentage

    model.eval()
    env = G2RLEnv(**env_config)
    num_actions = len(env.get_action_space())

    agent = DDQNAgent(model=model, action_space=list(range(num_actions)), device=device)

    success_rates, moving_costs, detour_rates = [], [], []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        target_idx = np.random.randint(env.num_agents)
        goal = tuple(env.goals[target_idx])
        state = obs[target_idx]
        opt_path = [state['global_xy']] + env.global_guidance[target_idx]
        agents = [agent if i == target_idx else AStarAgent() for i in range(env.num_agents)]

        success = False
        for t in range(50 + 10 * episode):
            actions = [a.act(o) for a, o in zip(agents, obs)]
            obs, _, terminated, _, _ = env.step(actions)
            pos = tuple(obs[target_idx]['global_xy'])
            if pos == goal:
                success = True
                moving_costs.append(moving_cost(t + 1, opt_path[0], opt_path[-1]))
                detour_rates.append(detour_percentage(t + 1, len(opt_path) - 1))
                break

        success_rates.append(1 if success else 0)

    return {
        'Success Rate': np.mean(success_rates),
        'Avg Moving Cost': np.mean(moving_costs) if moving_costs else float('inf'),
        'Avg Detour %': np.mean(detour_rates) if detour_rates else float('inf'),
    }


import matplotlib.pyplot as plt

def plot_training_results(logs, save_path=None):
    episodes = [log['Episode'] for log in logs]
    success = [log['Success'] for log in logs]
    moving_cost = [log['Moving Cost'] for log in logs]
    detour = [log['Detour Percentage'] for log in logs]
    loss = [log['Average Loss'] for log in logs]
    epsilon = [log['Average Epsilon'] for log in logs]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(episodes, success, label='Success')
    plt.title("Success per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Success")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(episodes, moving_cost, label='Moving Cost')
    plt.title("Moving Cost")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(episodes, detour, label='Detour %')
    plt.title("Detour Percentage")
    plt.xlabel("Episode")
    plt.ylabel("Detour %")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(episodes, loss, label='Loss')
    plt.plot(episodes, epsilon, label='Epsilon')
    plt.title("Loss & Epsilon")
    plt.xlabel("Episode")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"📈 图像保存至：{save_path}")
    plt.show()


def get_timestamp() -> str:
    now = datetime.now()
    timestamp = now.strftime('%H-%M-%d-%m-%Y')
    return timestamp


def get_normalized_probs(x: Union[List[float], None], size: int) -> np.ndarray:
    x = [1] * size if x is None else x + [0] * (size - len(x))
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def train(
        # model: torch.nn.Module,
        # scheduler: CurriculumScheduler,
        # map_settings: Dict[str, dict],
        # map_probs: Union[List[float], None],
        # num_episodes: int = 300,
        # batch_size: int = 32,
        # decay_range: int = 1000,
        # log_dir = 'logs',
        # lr: float = 0.001,
        # replay_buffer_size: int = 1000,
        # device: str = 'cuda'
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
        scheduler: Optional[CurriculumScheduler] = None,  # ← 可选+默认
        max_episode_seconds: int = 30
    ) -> DDQNAgent:
    timestamp = get_timestamp()
    writer = SummaryWriter(log_dir=Path(log_dir) / timestamp)
    # maps = [G2RLEnv(**args) for _, args in map_settings.items()]
    # maps = [G2RLEnv(**map_settings[name]) for name in map_settings]
    maps = []
    
    for name in map_settings:
        env = G2RLEnv(**map_settings[name])
        # print(f"✅ 初始化 env：{env}")
        # print(f"✅ env.reset: {env.reset}")
        maps.append(env)
    # map_probs = get_normalized_probs(map_probs, len(maps))
    map_probs = map_probs or [1.0 / len(maps)] * len(maps)
    agent = DDQNAgent(
        model,
        maps[0].get_action_space(),
        lr=lr,
        decay_range=decay_range,
        device=device,
        replay_buffer_size=replay_buffer_size,
    )

    pbar = tqdm(range(num_episodes), desc='Episodes', dynamic_ncols=True)
    
    
    # Curriculum Learning 阶段统计器
    episode = 0
    success_count = 0
    stage_success_count = 0
    stage_episode_count = 0

    # 记录每个阶段的评估结果


    episode_logs = []  # 👈 放在 train() 最前面



    # for episode in pbar:
    while scheduler.current_stage <= scheduler.max_stage:
        # scheduler.step(episode)


        # 手动地图选择（只训练该地图）
        # if episode == 0:
        # # ✅ 只在首个 episode 选择地图
        #     map_settings = scheduler.get_updated_map_settings()
        #     pbar.write(f"📶 当前训练阶段：Stage {scheduler.current_stage}")
        #     env, map_type = G2RLEnv.select_map_env(map_settings)
        #     pbar.write(f"🟢 首次选择地图：{map_type}")
        #     pbar.write(f"👥 当前 Agent 数量：{env.num_agents}")
            

        # else:
        #     pbar.write(f"🔁 继续使用地图：{map_type}")
        map_settings = map_settings
        map_type, cfg = next(iter(map_settings.items()))   # 直接取第一项
        env = G2RLEnv(**cfg)
        pbar.write(f"🟢 使用地图：{map_type}")
        maps = [env]
        # torch.save(model.state_dict(), f'models/{timestamp}.pt')
        stage = scheduler.current_stage
        num_agents = env.num_agents  # 当前环境中的 agent 数量
        model_name = f"models/g2rl_cl_stage{stage}_agents{num_agents}.pt"
        torch.save(model.state_dict(), model_name)
        # env, map_type = G2RLEnv.select_map_env(map_settings)
        # print(f"[DEBUG] env 类型: {type(env)}")
        # print(f"[DEBUG] env.reset 函数：{env.reset}")
        # print(f"✅ 当前 environment.py 路径: {G2RLEnv.__module__} 来自 {__import__('g2rl.environment').__file__}")
        
        obs, info = env.reset()
        pbar.write(f"🗺️ Episode {episode} using map: {map_type}")
        
        # print("✅ 是否有 goals？", hasattr(env, 'goals'))
        # print("✅ env.goals:", getattr(env, 'goals', None))
        target_idx = np.random.randint(env.num_agents)
        agents = [agent if i == target_idx else AStarAgent() for i in range(env.num_agents)]
        goal = tuple(env.goals[target_idx]) 
        state = obs[target_idx]
        # opt_path = [state['global_xy']] + env.global_guidance[target_idx]
         # 获取真实目标
        opt_path = [state['global_xy']] + env.global_guidance[target_idx]

        success_flag = False
        
        retrain_count = 0
        scalars = {
            'Reward': 0,
            'Moving Cost': 0,
            'Detour Percentage': 0,
            'Average Loss': 0,
            'Average Epsilon': 0,
        }

        timesteps_per_episode = 50 + 10 * episode
        episode_start_time = time.time()
        for timestep in range(timesteps_per_episode):
            if time.time() - episode_start_time > 30:
                pbar.write(f"⏰ Episode {episode} 超时（>30秒），强制终止本 episode")
                break
            actions = [agent.act(ob) for agent, ob in zip(agents, obs)]
            obs, reward, terminated, truncated, info = env.step(actions)
            # terminated[target_idx] = obs[target_idx]['global_xy'] == opt_path[-1]
            agent_pos = tuple(obs[target_idx]['global_xy'])
            
            terminated[target_idx] = agent_pos == goal
            # print(f"[EP {episode}] Agent pos: {agent_pos}, goal: {goal}, success: {terminated[target_idx]}")


            # if the target agent has finished or FOV does not contain the global guidance

            if terminated[target_idx]:
                success_flag = True
                scalars['Success'] = 1
                scalars['Moving Cost'] = moving_cost(timestep + 1, opt_path[0], opt_path[-1])
                scalars['Detour Percentage'] = detour_percentage(timestep + 1, len(opt_path) - 1)
                break
            
            # 提取状态特征矩阵
            if isinstance(state, dict) and 'obs' in state:
                obs_dict = state['obs']
            else:
                obs_dict = state

# 常见输入是 'view_cache'
            if 'view_cache' in obs_dict:
                state_obs = obs_dict['view_cache']  # shape 通常为 [T, H, W, C]
            else:
                raise ValueError(f"❌ state['obs'] 缺少 'view_cache'，目前内容为: {obs_dict.keys()}")

# 转换成 tensor，shape: [1, T, H, W, C] → 输入 CRNN
            state_tensor = torch.tensor(state_obs[..., :512], dtype=torch.float32).unsqueeze(0).to(device)

            agent.store(
                state,
                actions[target_idx],
                reward[target_idx],
                obs[target_idx],
                terminated[target_idx],
            )
            state = obs[target_idx]
            scalars['Reward'] += reward[target_idx]

            if len(agent.replay_buffer) >= batch_size:
                retrain_count += 1
                scalars['Average Loss'] += agent.retrain(batch_size)
                scalars['Average Epsilon'] += round(agent.epsilon, 4)

        
        # if success_flag:
        #     stage_success_count += scalars['Success']
        #     stage_episode_count += 1
        #     episode += 1
        # else:
        #     scalars['Success'] = 0  # ❌ 没有成功

        if not success_flag:
            scalars['Success'] = 0
        else:
            success_count += 1
            stage_success_count += 1

        stage_episode_count += 1
        episode += 1
        pbar.update(1)

        

        for name in scalars.keys():
            if 'Average' in name and retrain_count > 0:
                scalars[name] /= retrain_count

        # logging
        for name, value in scalars.items():
            writer.add_scalar(name, value, episode)
        # pbar.update(1)
        pbar.set_postfix(scalars)

        if scalars['Success'] == 1:
            pbar.write(f"[EP {episode}] ✅ 成功：Agent pos: {tuple(map(int, state['global_xy']))}, goal: {goal}")
        else:
            pbar.write(f"[EP {episode}] ❌ 失败：Agent pos: {tuple(map(int, state['global_xy']))}, goal: {goal}")
        pbar.write(f"⭐ 当前累计成功率：{success_count / (episode + 1) * 100:.2f}% ({success_count}/{episode + 1})")  #从训练开始到当前 episode 为止的总成功率 
        # pbar.write(f"📊 当前阶段成功率: {success_rate:.2f}")  #只统计当前 Curriculum Learning 阶段（Stage）中的成功率
        
                # ✅ 记录当前 episode 的训练统计信息
#         episode_logs.append({
#     'Episode': episode,
#     'Size': env.grid_config.size,
#     'Agents': env.num_agents,
#     'Density': env.grid_config.density,
#     'Success': scalars['Success'],
#     'Reward': scalars['Reward'],
#     'Moving Cost': scalars['Moving Cost'],
#     'Detour Percentage': scalars['Detour Percentage'],
#     'Average Loss': scalars['Average Loss'],
#     'Average Epsilon': scalars['Average Epsilon'],
# })
#         episode_logs.append({
#     'Episode': episode,
#     'Size': env.grid_config.size,
#     'Agents': env.num_agents,
#     'Density': env.grid_config.density,
#     'Success': scalars['Success'],
#     'Reward': scalars['Reward'],
#     'Moving Cost': scalars['Moving Cost'],
#     'Detour Percentage': scalars['Detour Percentage'],
#     'Average Loss': scalars['Average Loss'],
#     'Average Epsilon': scalars['Average Epsilon'],
#     'Complexity': predict_complexity(env)  # ✅ 新增
# })





        if stage_episode_count >= scheduler.episodes_per_stage:
            success_rate = stage_success_count / stage_episode_count
            # pbar.write(f"⭐ 当前累计成功率：{success_count / (episode + 1) * 100:.2f}% ({success_count}/{episode + 1})")  #从训练开始到当前 episode 为止的总成功率 
            
            previous_stage = scheduler.current_stage
            scheduler.update(success_rate, pbar)
            

            user_input = input(f"🚦 Stage {scheduler.current_stage - 1} 完成。是否继续训练？(y/n): ")
            if user_input.lower() != 'y':
                pbar.write("🛑 用户选择终止训练")
                torch.save(model.state_dict(), f'models/stage{scheduler.current_stage - 1}_final.pt')
                plot_training_results(episode_logs, save_path='training_plot.png')
                writer.close()
                return agent
            # 重置阶段内计数器
            stage_success_count = 0
            stage_episode_count = 0


            if scheduler.current_stage == 4:
                pbar.write("🛑 已完成 Stage 3，训练自动终止")
                torch.save(model.state_dict(), f'models/stage3_final.pt')
                writer.close()
                return agent
    

            if scheduler.current_stage != previous_stage:
                map_settings = scheduler.get_updated_map_settings()
                env, map_type = G2RLEnv.select_map_env(map_settings)
                pbar.write(f"🆕 晋级后重新选择地图：{map_type}")
                pbar.write(f"👥 当前 Agent 数量：{env.num_agents}")


            if scheduler.current_stage > scheduler.max_stage:
                pbar.write("🎉 所有阶段完成，训练结束！")
                break

    # plot_training_results(episode_logs, save_path='training_plot.png')
    # df_logs = pd.DataFrame(episode_logs)
    # df_logs.to_csv('logs/episode_logs.csv', index=False, encoding='utf-8-sig')
    # pbar.write("📄 episode_logs 已保存为 logs/episode_logs.csv")

    writer.close()
    return agent


# def train(
#         model: torch.nn.Module,
#         map_settings: Dict[str, dict],
#         map_probs: Union[List[float], None],
#         num_episodes: int = 300,
#         batch_size: int = 32,
#         decay_range: int = 1000,
#         log_dir='logs',
#         lr: float = 0.001,
#         replay_buffer_size: int = 1000,
#         device: str = 'cuda',
#         scheduler: Optional[CurriculumScheduler] = None,  # 自动晋级/重复用
#         max_episode_seconds: int = 30
#     ) -> DDQNAgent:

#     timestamp = get_timestamp()
#     writer = SummaryWriter(log_dir=Path(log_dir) / timestamp)

#     # 环境/agent
#     maps = []
#     for name in map_settings:
#         env = G2RLEnv(**map_settings[name])
#         maps.append(env)
#     map_probs = map_probs or [1.0 / len(maps)] * len(maps)

#     # 用第一个 env 的动作空间初始化 agent
#     agent = DDQNAgent(
#         model,
#         maps[0].get_action_space(),
#         lr=lr,
#         decay_range=decay_range,
#         device=device,
#         replay_buffer_size=replay_buffer_size,
#     )

#     pbar = tqdm(range(num_episodes), desc='Episodes', dynamic_ncols=True)

#     # 训练计数器
#     episode = 0
#     success_count_total = 0

#     # 阶段计数器（用于“到达阶段上限但未达标→重复”）
#     stage_success_count = 0
#     stage_episode_count = 0

#     # —— 阈值：优先用 scheduler.threshold，否则回退到 0.8
#     stage_threshold = getattr(scheduler, "threshold", 0.8)

#     # 主循环：直到所有阶段完成或达到总 episodes 上限
#     while (scheduler is None) or (scheduler.current_stage <= scheduler.max_stage):
#         if episode >= num_episodes:
#             break

#         # 1) 获取“当前阶段”的地图配置，重建 env
#         cur_map_cfg = scheduler.get_updated_map_settings() if scheduler else map_settings
#         map_type, cfg = next(iter(cur_map_cfg.items()))
#         env = G2RLEnv(**cfg)
#         pbar.write(f"🟢 使用地图：{map_type} | Stage {scheduler.current_stage if scheduler else '-'} | Agents={env.num_agents}")

#         # 2) reset & 准备一集
#         obs, info = env.reset()
#         target_idx = np.random.randint(env.num_agents)
#         agents = [agent if i == target_idx else AStarAgent() for i in range(env.num_agents)]
#         goal = tuple(env.goals[target_idx])
#         state = obs[target_idx]
#         opt_path = [state['global_xy']] + env.global_guidance[target_idx]

#         success_flag = False
#         retrain_count = 0
#         scalars = {
#             'Reward': 0.0,
#             'Moving Cost': 0.0,
#             'Detour Percentage': 0.0,
#             'Average Loss': 0.0,
#             'Average Epsilon': 0.0,
#             'Success': 0
#         }

#         # 3) 跑一集
#         timesteps_per_episode = 50 + 10 * episode
#         episode_start_time = time.time()

#         for timestep in range(timesteps_per_episode):
#             # 超时保护
#             if time.time() - episode_start_time > max_episode_seconds:
#                 pbar.write(f"⏰ Episode {episode} 超时（>{max_episode_seconds}s），强制终止本 episode")
#                 break

#             actions = [ag.act(o) for ag, o in zip(agents, obs)]
#             obs, reward, terminated, truncated, info = env.step(actions)

#             # 到达目标判定
#             agent_pos = tuple(obs[target_idx]['global_xy'])
#             done = (agent_pos == goal)
#             terminated[target_idx] = done

#             if done:
#                 success_flag = True
#                 scalars['Success'] = 1
#                 scalars['Moving Cost'] = moving_cost(timestep + 1, opt_path[0], opt_path[-1])
#                 scalars['Detour Percentage'] = detour_percentage(timestep + 1, len(opt_path) - 1)
#                 break

#             # 采样&学习
#             # —— 你的状态提取逻辑（尽量使用 obs[target_idx]，不要用旧 state 里的 'obs'）——
#             # 这里保留你的原实现，注意 state 的更新放在最后
#             agent.store(
#                 state,
#                 actions[target_idx],
#                 reward[target_idx],
#                 obs[target_idx],
#                 terminated[target_idx],
#             )
#             state = obs[target_idx]
#             scalars['Reward'] += float(reward[target_idx])

#             if len(agent.replay_buffer) >= batch_size:
#                 retrain_count += 1
#                 scalars['Average Loss'] += float(agent.retrain(batch_size))
#                 scalars['Average Epsilon'] += float(agent.epsilon)

#         # 4) 统计本集
#         if retrain_count > 0:
#             scalars['Average Loss'] /= retrain_count
#             scalars['Average Epsilon'] /= retrain_count

#         if success_flag:
#             success_count_total += 1
#             stage_success_count += 1
#         scalars['Success'] = 1 if success_flag else 0

#         stage_episode_count += 1
#         episode += 1
#         pbar.update(1)

#         # logging
#         for name, value in scalars.items():
#             writer.add_scalar(name, value, episode)

#         pbar.set_postfix(
#             Stage=(scheduler.current_stage if scheduler else "-"),
#             SR_total=f"{success_count_total / max(1, episode):.2f}",
#             R=f"{scalars['Reward']:.2f}",
#         )

#         # 5) 把结果写进 scheduler 的滑窗，并判定晋级/重复
#         if scheduler is not None:
#             scheduler.add_episode_result(scalars['Success'])
#             window_sr = scheduler.current_window_sr()

#             # 到达阶段上限：根据阶段成功率（而非滑窗）做一次硬判定
#             if stage_episode_count >= scheduler.episodes_per_stage:
#                 stage_sr = stage_success_count / max(1, stage_episode_count)
#                 if stage_sr >= stage_threshold:
#                     scheduler.advance(pbar)
#                     # 重置阶段计数器
#                     stage_success_count = 0
#                     stage_episode_count = 0
#                     # 最后一阶段且已达标可早停
#                     if scheduler.is_done():
#                         break
#                     # 进入下一循环会按新阶段配置重建 env
#                     continue
#                 else:
#                     # 未达标：重复当前 stage
#                     scheduler.repeat_stage(pbar)
#                     stage_success_count = 0
#                     stage_episode_count = 0
#                     # 继续在当前 stage 训练
#                     continue

#             # 可选：如果你更偏好“滑窗达标立刻晋级”，也保留这条快速通道
#             if scheduler.ready_to_advance():
#                 scheduler.advance(pbar)
#                 stage_success_count = 0
#                 stage_episode_count = 0
#                 if scheduler.is_done():
#                     break
#                 continue

#     # 结束：记录最终成功率（用滑窗或全局）
#     # ✅ 只用全局：所有 episode 成功数 / 所有 episode 数
#     final_sr = success_count_total / max(1, episode)
#     agent.final_success_rate = float(final_sr)

# # （可选）调试一下，确认口径
#     print(f"[train] final_sr(global) = {success_count_total}/{episode} = {agent.final_success_rate:.6f}")


#     writer.close()
#     return agent


    

if __name__ == '__main__':
    import os
    from collections import deque  # 避免 deque 标黄
    import yaml
    import torch
    import pandas as pd
    import numpy as np

    # 确保目录存在
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 1) 读取地图配置（你的生成文件）
    MAP_SETTINGS_PATH = 'C:/Users/MSc_SEIoT_1/MAPF_G2RL-main/g2rl/map_settings_generated.yaml'
    with open(MAP_SETTINGS_PATH, "r", encoding="utf-8") as f:
        base_map_settings = yaml.safe_load(f)

    # 顶层为 list 时，转成 {name: spec}
    if isinstance(base_map_settings, list):
        base_map_settings = { (m.get("name") or f"map_{i}"): m for i, m in enumerate(base_map_settings) }

    # 2) 构建基于 complexity 的 scheduler（用你前面贴的 ComplexityScheduler）
    scheduler = ComplexityScheduler(
        base_map_settings=base_map_settings,
        n_stages=5,
        min_per_stage=10,
        episodes_per_stage=100,
        threshold=0.70,
        window_size=100,
        shuffle_each_stage=True,
        seed=0,
        size_mode="max",  # 和你训练公式的 Size 定义保持一致
    )

    # 3) 设备 & 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNNModel().to(device)

    # 4) 训练
    trained_agent = train(
        model=model,
        scheduler=scheduler,                           # ✅ 用新的 scheduler
        map_settings=scheduler.get_updated_map_settings(),  # 初始一张；train 内每集会再取
        map_probs=None,
        num_episodes=300,
        batch_size=32,
        replay_buffer_size=500,
        decay_range=10_000,
        log_dir='logs',
        device=device,
        # 如你的 train 签名里还有 max_episode_seconds 等参数，也一起传：
        # max_episode_seconds=30,
    )

    # 5) 保存模型
    torch.save(model.state_dict(), 'models/best_model.pt')
    print('✅ 模型已保存到 models/best_model.pt')


