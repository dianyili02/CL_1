# traincl.py  — cleaned & fixed
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional, Union
from collections import deque
import os
import sys
import time
import csv
import random
import math
import inspect
# 放在文件顶部其它 import 之后
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml

from pogema import AStarAgent

# --- 项目根路径确保在 sys.path 中 ---
project_root = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 你项目的模块 ---
from g2rl.environment import G2RLEnv
from g2rl.agent import DDQNAgent
from g2rl.network import CRNNModel
from g2rl import moving_cost, detour_percentage

# 如果仍需原 Scheduler，可保留；当前脚本使用 ComplexityScheduler
# from g2rl.curriculum import CurriculumScheduler

# ============ 安全构造器：过滤 __init__ 不认识的键 ============
def build_env_from_raw(raw_cfg: dict) -> G2RLEnv:
    """
    只把 G2RLEnv.__init__ 认识的参数传进去；
    其他（grid/starts/goals 等）在实例化后作为属性挂载。
    """
    sig = inspect.signature(G2RLEnv.__init__)
    allowed = {
        p.name for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    allowed.discard("self")

    # 如有命名不一致，在这里做重命名映射（按你项目需要补）
    rename = {
        # 'size': 'map_size',
        # 'num_agents': 'n_agents',
    }

    ctor_cfg = {}
    for k, v in raw_cfg.items():
        kk = rename.get(k, k)
        if kk in allowed:
            ctor_cfg[kk] = v

    env = G2RLEnv(**ctor_cfg)

    # 把额外信息挂到 env 上（不进 __init__）
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

# ============ Complexity-based Curriculum Scheduler ============
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
FEATURE_MEAN_STD = None  # 若训练时做过标准化，按 {'Size':(mu, sigma), ...} 传入


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
def _compute_complexities_for_settings(base_map_settings: Dict[str, dict],
                                       size_mode: str = "max") -> pd.DataFrame:
    rows = []
    for name, spec in base_map_settings.items():
        try:
            cpx, used, raw = compute_map_complexity(
                spec, intercept=INTERCEPT, weights=WEIGHTS,
                feature_mean_std=FEATURE_MEAN_STD, size_mode=size_mode
            )
            # ★ 在 spec 里塞入 complexity
            spec_with_cpx = dict(spec)
            spec_with_cpx["complexity"] = float(cpx)

            rows.append({
                "name": name,
                "complexity": float(cpx),
                "spec": spec_with_cpx,   # ★ 用带 complexity 的 spec
            })
        except Exception as e:
            rows.append({"name": name, "error": str(e), "spec": spec})
    df = pd.DataFrame(rows)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    return df.sort_values("complexity").reset_index(drop=True)

def _rolling_mean(x, w=50):
    if len(x) == 0:
        return np.array([])
    w = max(1, int(w))
    c = np.cumsum(np.insert(x, 0, 0))
    # 简单滚动平均；对前 w-1 项用更短窗口避免空缺
    rm = (c[w:] - c[:-w]) / float(w)
    head = [np.mean(x[:i+1]) for i in range(min(w-1, len(x)))]
    return np.array(head + rm.tolist())

def make_training_plots(out_dir: str, df: pd.DataFrame, *, win: int = 50):
    os.makedirs(out_dir, exist_ok=True)

    # -------- 1) Success & Rolling SR vs Episode --------
    plt.figure(figsize=(10, 5))
    ep = df["episode"].values
    succ = df["success"].values.astype(float)
    plt.plot(ep, succ, label="Success (0/1)", linewidth=1)
    rm = _rolling_mean(succ, w=win)
    if len(rm) > 0:
        plt.plot(ep[:len(rm)], rm, linewidth=2, label=f"Rolling SR (w={win})")
    plt.xlabel("Episode"); plt.ylabel("Success / SR")
    plt.title("Success & Rolling Success-Rate")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "sr_curve.png"), dpi=150); plt.close()

    # -------- 2) Loss & Epsilon vs Episode --------
    if "avg_loss" in df.columns or "avg_epsilon" in df.columns:
        plt.figure(figsize=(10, 5))
        if "avg_loss" in df.columns:
            plt.plot(df["episode"], df["avg_loss"], label="Avg Loss")
        if "avg_epsilon" in df.columns:
            plt.plot(df["episode"], df["avg_epsilon"], label="Avg Epsilon")
        plt.xlabel("Episode"); plt.title("Loss & Epsilon")
        plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "loss_epsilon.png"), dpi=150); plt.close()

    # -------- 3) Steps / MovingCost / Detour vs Episode --------
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(3,1,1); ax1.plot(df["episode"], df["steps"]); ax1.set_title("Steps per Episode"); ax1.grid(True, alpha=0.3)
    ax2 = plt.subplot(3,1,2); 
    if "moving_cost" in df.columns:
        ax2.plot(df["episode"], df["moving_cost"]); ax2.set_title("Moving Cost (success only may be non-NaN)"); ax2.grid(True, alpha=0.3)
    ax3 = plt.subplot(3,1,3);
    if "detour_pct" in df.columns:
        ax3.plot(df["episode"], df["detour_pct"]); ax3.set_title("Detour Percentage (success only may be non-NaN)"); ax3.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "steps_moving_detour.png"), dpi=150); plt.close()

    # -------- 4) Complexity vs Success (per-episode scatter) --------
    if "complexity" in df.columns and df["complexity"].notna().any():
        plt.figure(figsize=(8,5))
        plt.scatter(df["complexity"], df["success"], s=18)
        plt.xlabel("Complexity"); plt.ylabel("Success (0/1)")
        plt.title("Episode Success vs Complexity")
        plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "success_vs_complexity_scatter.png"), dpi=150); plt.close()

    # -------- 5) Success by Complexity Bucket --------
    if "complexity" in df.columns and df["complexity"].notna().any():
        d = df.dropna(subset=["complexity"]).copy()
        if len(d) > 3:
            edges = np.quantile(d["complexity"].values, np.linspace(0, 1, 6))  # 5 桶
            d["bucket"] = pd.cut(d["complexity"], bins=edges, include_lowest=True, right=False)
            sr_by_bucket = d.groupby("bucket")["success"].mean()
            plt.figure(figsize=(8,5)); sr_by_bucket.plot(kind="bar")
            plt.ylabel("Mean Success Rate"); plt.title("Success Rate by Complexity Bucket (episodes)")
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, "sr_by_bucket.png"), dpi=150); plt.close()

    # -------- 6) Per-Stage SR --------
    if "stage" in df.columns:
        st = df.groupby("stage")["success"].mean()
        plt.figure(figsize=(8,5)); st.plot(kind="bar")
        plt.ylabel("Mean Success Rate"); plt.title("Success Rate by Stage (episodes)")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "sr_by_stage.png"), dpi=150); plt.close()


class ComplexityScheduler:
    """
    分阶段课程：
      - 每个阶段持有一组 maps（由 complexity 分位切分）
      - 每阶段至少跑 min_episodes_per_stage 集
      - 达到阈值（滑窗或阶段累计）后晋级
    """

    
    def __init__(self,
                 base_map_settings: Dict[str, dict],
                 n_stages: int = 5,
                 min_per_stage: int = 5,
                 # 判定相关
                 min_episodes_per_stage: int =200 ,   # 每阶段至少训练这么多 episode
                 threshold: float = 0.70,             # 成功率阈值
                 window_size: int = 100,              # 滑窗大小
                 use_window_sr: bool = True,          # True=用滑窗SR；False=用阶段累计SR
                 # 其它
                 shuffle_each_stage: bool = True,
                 seed: int = 0,
                 size_mode: str = "max"):
        self.min_episodes_per_stage = int(min_episodes_per_stage)
        self.threshold = float(threshold)
        self.window_size = int(window_size)
        self.use_window_sr = bool(use_window_sr)

        self._rng = random.Random(seed)
        df = _compute_complexities_for_settings(base_map_settings, size_mode=size_mode)
        stages = _build_stages_by_quantile_df(df, n_stages=n_stages, min_per_stage=min_per_stage)

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

        # 轮转采样索引
        self._idx_in_stage = 0
        # 统计
        self._win = deque(maxlen=self.window_size)  # 滑窗
        self._ep_in_stage = 0                       # 阶段 episode 数
        self._succ_in_stage = 0                     # 阶段成功 episode 数
    
    

    # ============ 取图 ============
    def get_updated_map_settings(self) -> Dict[str, dict]:
        if self.current_stage > self.max_stage:
            return {}
        items = self._stage_items[self.current_stage]
        if not items:
            raise RuntimeError(f"Stage {self.current_stage} 没有地图。")
        item = items[self._idx_in_stage]
        self._idx_in_stage = (self._idx_in_stage + 1) % len(items)
        # spec 中保留 complexity，方便训练侧打印
        return {item["name"]: item["spec"]}

    # ============ 统计 & 判定 ============
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
        """是否满足晋级条件：跑满最少集数 且 成功率达标"""
        if self._ep_in_stage < self.min_episodes_per_stage:
            return False
        sr = self.window_sr() if self.use_window_sr else self.stage_sr()
        return sr >= self.threshold

    # ============ 阶段切换 ============
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


# ================== 训练相关 ==================
def get_timestamp() -> str:
    return datetime.now().strftime('%H-%M-%d-%m-%Y')

def get_normalized_probs(x: Union[List[float], None], size: int) -> np.ndarray:
    x = [1] * size if x is None else x + [0] * (size - len(x))
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def train(
        model: torch.nn.Module,
        map_settings: Dict[str, dict],
        map_probs: Union[List[float], None],
        num_episodes: int = 300,
        batch_size: int = 32,
        decay_range: int = 1000,
        log_dir: str = 'logs',
        lr: float = 0.001,
        replay_buffer_size: int = 1000,
        device: str = 'cuda',
        scheduler: Optional[ComplexityScheduler] = None,
        max_episode_seconds: int = 30,
        run_dir: Optional[str] = None
    ) -> DDQNAgent:

    # === 统一确定输出目录 ===
    if run_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%H-%M-%d-%m-%Y')
        run_dir = Path(log_dir) / timestamp
    else:
        run_dir = Path(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard 写日志到 run_dir
    writer = SummaryWriter(log_dir=run_dir)

    training_logs = []



    # 初始化第一个 env 获取动作空间
    first_name = next(iter(map_settings))
    first_env = build_env_from_raw(map_settings[first_name])
    agent = DDQNAgent(
        model,
        first_env.get_action_space(),
        lr=lr,
        decay_range=decay_range,
        device=device,
        replay_buffer_size=replay_buffer_size,
    )

    pbar = tqdm(range(num_episodes), desc='Episodes', dynamic_ncols=True)

    episode = 0
    success_count_total = 0
    stage_success_count = 0
    stage_episode_count = 0
    stage_threshold = getattr(scheduler, "threshold", 0.8) if scheduler else 0.8

    while (scheduler is None) or (scheduler.current_stage <= scheduler.max_stage):
        if episode >= num_episodes:
            break

        # 1) 阶段地图
        cur_map_cfg = scheduler.get_updated_map_settings() if scheduler else map_settings
        map_type, cfg = next(iter(cur_map_cfg.items()))
        env = build_env_from_raw(cfg)

        cpx_val = cfg.get("complexity", None)  # ★ 直接从 cfg 里取
        stage_id = scheduler.current_stage if scheduler else '-'
        if cpx_val is not None:
            pbar.write(f"🟢 使用地图：{map_type} | Stage {stage_id} | Agents={env.num_agents} | Complexity={cpx_val:.3f}")
        else:
            pbar.write(f"🟢 使用地图：{map_type} | Stage {stage_id} | Agents={env.num_agents}")



        # 2) reset
        obs, info = env.reset()
        target_idx = np.random.randint(env.num_agents)
        agents = [agent if i == target_idx else AStarAgent() for i in range(env.num_agents)]
        goal = tuple(env.goals[target_idx])
        state = obs[target_idx]
        opt_path = [state['global_xy']] + env.global_guidance[target_idx]

        success_flag = False
        retrain_count = 0
        scalars = {
            'Reward': 0.0,
            'Moving Cost': 0.0,
            'Detour Percentage': 0.0,
            'Average Loss': 0.0,
            'Average Epsilon': 0.0,
            'Success': 0
        }

        # 3) 一集
        timesteps_per_episode = 50 + 10 * episode
        episode_start_time = time.time()

        for t in range(timesteps_per_episode):
            if time.time() - episode_start_time > max_episode_seconds:
                pbar.write(f"⏰ Episode {episode} 超时（>{max_episode_seconds}s），强制终止")
                break

            actions = [ag.act(o) for ag, o in zip(agents, obs)]
            obs, reward, terminated, truncated, info = env.step(actions)

            agent_pos = tuple(obs[target_idx]['global_xy'])
            done = (agent_pos == goal)
            terminated[target_idx] = done

            if done:
                success_flag = True
                scalars['Success'] = 1
                scalars['Moving Cost'] = moving_cost(t + 1, opt_path[0], opt_path[-1])
                scalars['Detour Percentage'] = detour_percentage(t + 1, len(opt_path) - 1)
                break

            # 经验
            agent.store(
                state,
                actions[target_idx],
                reward[target_idx],
                obs[target_idx],
                terminated[target_idx],
            )
            state = obs[target_idx]
            scalars['Reward'] += float(reward[target_idx])

            if len(agent.replay_buffer) >= batch_size:
                retrain_count += 1
                scalars['Average Loss'] += float(agent.retrain(batch_size))
                scalars['Average Epsilon'] += float(agent.epsilon)

        if retrain_count > 0:
            scalars['Average Loss'] /= retrain_count
            scalars['Average Epsilon'] /= retrain_count

        if success_flag:
            success_count_total += 1
            stage_success_count += 1
        scalars['Success'] = 1 if success_flag else 0
                # 统计步数
        steps_this = t + 1 if 't' in locals() else 0

        # 记录一条 episodio 日志
        training_logs.append({
            "episode": episode,
            "stage": (scheduler.current_stage if scheduler else -1),
            "map": map_type,
            "agents": env.num_agents,
            "complexity": (cpx_val if cpx_val is not None else np.nan),
            "success": int(scalars['Success']),
            "reward": float(scalars['Reward']),
            "steps": int(steps_this),
            "avg_loss": float(scalars['Average Loss']),
            "avg_epsilon": float(scalars['Average Epsilon']),
            "moving_cost": float(scalars.get('Moving Cost', np.nan)),
            "detour_pct": float(scalars.get('Detour Percentage', np.nan)),
        })

        stage_episode_count += 1
        episode += 1
        pbar.update(1)

        for name, value in scalars.items():
            writer.add_scalar(name, value, episode)

        pbar.set_postfix(
            Stage=(scheduler.current_stage if scheduler else "-"),
            SR_total=f"{success_count_total / max(1, episode):.2f}",
            R=f"{scalars['Reward']:.2f}",
        )

        # 5) 课程逻辑
        # 5) 课程逻辑：达标晋级（跑满最少集数 + 成功率达标）
        if scheduler is not None:
            scheduler.add_episode_result(scalars['Success'])

            if scheduler.should_advance():
                scheduler.advance(pbar)
                # 进入下一阶段就继续循环（会自动抽取下一阶段的更难地图）
                if scheduler.is_done():
                    break
                continue
            else:
                # 没达标但还没满最少集数：继续训练本阶段
                # 如果你想加“满最少集数但还未达标→强制重置并重复本阶段”，可加：
                if scheduler._ep_in_stage >= scheduler.min_episodes_per_stage:
                    scheduler.repeat_stage(pbar)
                # 然后继续该阶段训练


        #     # 可选：滑窗快速通道
        # elif scheduler.ready_to_advance():
        #         scheduler.advance(pbar)
        #         stage_success_count = 0
        #         stage_episode_count = 0
        #         if scheduler.is_done():
        #             break

    final_sr = success_count_total / max(1, episode)
    agent.final_success_rate = float(final_sr)
    print(f"[train] final_sr(global) = {success_count_total}/{episode} = {agent.final_success_rate:.6f}")
    
        # === 保存 CSV & 可视化 ===
    df_train = pd.DataFrame(training_logs)
    csv_path = run_dir / "episodes.csv"
    df_train.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"📝 训练日志已保存：{csv_path}")

    try:
        make_training_plots(str(run_dir), df_train, win=50)
        print(f"📊 训练可视化已保存到：{run_dir}")
    except Exception as e:
        print(f"⚠️ 生成可视化失败：{e}")


    writer.close()
    return agent

# ================== 入口 ==================
if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    MAP_SETTINGS_PATH = 'C:/Users/MSc_SEIoT_1/MAPF_G2RL-main/g2rl/map_settings_generated.yaml'
    with open(MAP_SETTINGS_PATH, "r", encoding="utf-8") as f:
        base_map_settings = yaml.safe_load(f)

    if isinstance(base_map_settings, list):
        base_map_settings = { (m.get("name") or f"map_{i}"): m for i, m in enumerate(base_map_settings) }

    scheduler = ComplexityScheduler(
        base_map_settings=base_map_settings,
        n_stages=5,
        min_per_stage=10,
        min_episodes_per_stage=100,
        threshold=0.70,
        window_size=100,
        shuffle_each_stage=True,
        seed=0,
        size_mode="max",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNNModel().to(device)

    trained_agent = train(
        model=model,
        scheduler=scheduler,
        map_settings=scheduler.get_updated_map_settings(),  # 初始一张；train 内每集会再取
        map_probs=None,
        num_episodes=300,
        batch_size=32,
        replay_buffer_size=500,
        decay_range=10_000,
        log_dir='logs',
        device=device,
        run_dir="C:/Users/MSc_SEIoT_1/MAPF_G2RL-main/pics/traincl"
    )

    torch.save(model.state_dict(), 'models/model1.pt')
    print('✅ 模型已保存到 models/model1.pt')
