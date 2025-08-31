# # traincl.py  — cleaned & fixed
# from pathlib import Path
# from datetime import datetime
# from tqdm import tqdm
# from typing import List, Dict, Optional, Union
# from collections import deque
# import os
# import sys
# import time
# import csv
# import random
# import math
# import inspect
# # 放在文件顶部其它 import 之后
# import matplotlib
# import matplotlib.pyplot as plt

# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.tensorboard import SummaryWriter
# import yaml

# from pogema import AStarAgent

# # --- 项目根路径确保在 sys.path 中 ---
# project_root = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main"
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# # --- 你项目的模块 ---
# from g2rl.environment import G2RLEnv
# from g2rl.agent import DDQNAgent
# from g2rl.network import CRNNModel
# from g2rl import moving_cost, detour_percentage

# # 如果仍需原 Scheduler，可保留；当前脚本使用 ComplexityScheduler
# # from g2rl.curriculum import CurriculumScheduler

# # ============ 安全构造器：过滤 __init__ 不认识的键 ============
# def build_env_from_raw(raw_cfg: dict) -> G2RLEnv:
#     """
#     只把 G2RLEnv.__init__ 认识的参数传进去；
#     其他（grid/starts/goals 等）在实例化后作为属性挂载。
#     """
#     sig = inspect.signature(G2RLEnv.__init__)
#     allowed = {
#         p.name for p in sig.parameters.values()
#         if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
#     }
#     allowed.discard("self")

#     # 如有命名不一致，在这里做重命名映射（按你项目需要补）
#     rename = {
#         # 'size': 'map_size',
#         # 'num_agents': 'n_agents',
#     }

#     ctor_cfg = {}
#     for k, v in raw_cfg.items():
#         kk = rename.get(k, k)
#         if kk in allowed:
#             ctor_cfg[kk] = v

#     env = G2RLEnv(**ctor_cfg)

#     # 把额外信息挂到 env 上（不进 __init__）
#     if "grid" in raw_cfg:
#         try:
#             env.grid = (np.array(raw_cfg["grid"]) > 0).astype(np.uint8)
#         except Exception:
#             env.grid = None
#     if "starts" in raw_cfg:
#         env.starts = raw_cfg["starts"]
#     if "goals" in raw_cfg:
#         env.goals = raw_cfg["goals"]

#     return env

# # ============ Complexity-based Curriculum Scheduler ============
# from g2rl.complexity_module import compute_map_complexity

# INTERCEPT = 0.848
# WEIGHTS = {
#     'Size': 0.021,
#     'Agents': -0.010,
#     'Density': 0.077,
#     'Density_actual': 0.132,
#     'LDD': 0.027,
#     'BN': -0.128,
#     'MC': 0.039,
#     'DLR': 0.002,
# }
# FEATURE_MEAN_STD = None  # 若训练时做过标准化，按 {'Size':(mu, sigma), ...} 传入


# def _build_stages_by_quantile_df(df: pd.DataFrame, n_stages: int = 5, min_per_stage: int = 5):
#     if len(df) == 0:
#         raise ValueError("没有可用地图用于 complexity 课程。")
#     qs = np.linspace(0, 1, n_stages + 1)
#     edges = np.quantile(df["complexity"].values, qs)
#     stages = []
#     for i in range(n_stages):
#         lo, hi = float(edges[i]), float(edges[i+1]) + 1e-12
#         sub = df[(df["complexity"] >= lo) & (df["complexity"] < hi)]
#         if len(sub) < min_per_stage:
#             need = min_per_stage - len(sub)
#             center = (lo + hi) / 2.0
#             extra = df.iloc[(df["complexity"] - center).abs().argsort()[:need]]
#             sub = pd.concat([sub, extra]).drop_duplicates(subset=["name"])
#         stages.append({
#             "stage": i,
#             "cpx_min": lo,
#             "cpx_max": hi,
#             "items": sub.to_dict("records"),  # 每条含 name/spec/complexity
#         })
#     return stages
# def _compute_complexities_for_settings(base_map_settings: Dict[str, dict],
#                                        size_mode: str = "max") -> pd.DataFrame:
#     rows = []
#     for name, spec in base_map_settings.items():
#         try:
#             cpx, used, raw = compute_map_complexity(
#                 spec, intercept=INTERCEPT, weights=WEIGHTS,
#                 feature_mean_std=FEATURE_MEAN_STD, size_mode=size_mode
#             )
#             # ★ 在 spec 里塞入 complexity
#             spec_with_cpx = dict(spec)
#             spec_with_cpx["complexity"] = float(cpx)

#             rows.append({
#                 "name": name,
#                 "complexity": float(cpx),
#                 "spec": spec_with_cpx,   # ★ 用带 complexity 的 spec
#             })
#         except Exception as e:
#             rows.append({"name": name, "error": str(e), "spec": spec})
#     df = pd.DataFrame(rows)
#     if "error" in df.columns:
#         df = df[df["error"].isna()]
#     return df.sort_values("complexity").reset_index(drop=True)

# def _rolling_mean(x, w=50):
#     if len(x) == 0:
#         return np.array([])
#     w = max(1, int(w))
#     c = np.cumsum(np.insert(x, 0, 0))
#     # 简单滚动平均；对前 w-1 项用更短窗口避免空缺
#     rm = (c[w:] - c[:-w]) / float(w)
#     head = [np.mean(x[:i+1]) for i in range(min(w-1, len(x)))]
#     return np.array(head + rm.tolist())

# def make_training_plots(out_dir: str, df: pd.DataFrame, *, win: int = 50):
#     os.makedirs(out_dir, exist_ok=True)

#     # -------- 1) Success & Rolling SR vs Episode --------
#     plt.figure(figsize=(10, 5))
#     ep = df["episode"].values
#     succ = df["success"].values.astype(float)
#     plt.plot(ep, succ, label="Success (0/1)", linewidth=1)
#     rm = _rolling_mean(succ, w=win)
#     if len(rm) > 0:
#         plt.plot(ep[:len(rm)], rm, linewidth=2, label=f"Rolling SR (w={win})")
#     plt.xlabel("Episode"); plt.ylabel("Success / SR")
#     plt.title("Success & Rolling Success-Rate")
#     plt.grid(True, alpha=0.3); plt.legend()
#     plt.tight_layout(); plt.savefig(os.path.join(out_dir, "sr_curve.png"), dpi=150); plt.close()

#     # -------- 2) Loss & Epsilon vs Episode --------
#     if "avg_loss" in df.columns or "avg_epsilon" in df.columns:
#         plt.figure(figsize=(10, 5))
#         if "avg_loss" in df.columns:
#             plt.plot(df["episode"], df["avg_loss"], label="Avg Loss")
#         if "avg_epsilon" in df.columns:
#             plt.plot(df["episode"], df["avg_epsilon"], label="Avg Epsilon")
#         plt.xlabel("Episode"); plt.title("Loss & Epsilon")
#         plt.grid(True, alpha=0.3); plt.legend()
#         plt.tight_layout(); plt.savefig(os.path.join(out_dir, "loss_epsilon.png"), dpi=150); plt.close()

#     # -------- 3) Steps / MovingCost / Detour vs Episode --------
#     plt.figure(figsize=(12, 6))
#     ax1 = plt.subplot(3,1,1); ax1.plot(df["episode"], df["steps"]); ax1.set_title("Steps per Episode"); ax1.grid(True, alpha=0.3)
#     ax2 = plt.subplot(3,1,2); 
#     if "moving_cost" in df.columns:
#         ax2.plot(df["episode"], df["moving_cost"]); ax2.set_title("Moving Cost (success only may be non-NaN)"); ax2.grid(True, alpha=0.3)
#     ax3 = plt.subplot(3,1,3);
#     if "detour_pct" in df.columns:
#         ax3.plot(df["episode"], df["detour_pct"]); ax3.set_title("Detour Percentage (success only may be non-NaN)"); ax3.grid(True, alpha=0.3)
#     plt.tight_layout(); plt.savefig(os.path.join(out_dir, "steps_moving_detour.png"), dpi=150); plt.close()

#     # -------- 4) Complexity vs Success (per-episode scatter) --------
#     if "complexity" in df.columns and df["complexity"].notna().any():
#         plt.figure(figsize=(8,5))
#         plt.scatter(df["complexity"], df["success"], s=18)
#         plt.xlabel("Complexity"); plt.ylabel("Success (0/1)")
#         plt.title("Episode Success vs Complexity")
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout(); plt.savefig(os.path.join(out_dir, "success_vs_complexity_scatter.png"), dpi=150); plt.close()

#     # -------- 5) Success by Complexity Bucket --------
#     if "complexity" in df.columns and df["complexity"].notna().any():
#         d = df.dropna(subset=["complexity"]).copy()
#         if len(d) > 3:
#             edges = np.quantile(d["complexity"].values, np.linspace(0, 1, 6))  # 5 桶
#             d["bucket"] = pd.cut(d["complexity"], bins=edges, include_lowest=True, right=False)
#             sr_by_bucket = d.groupby("bucket")["success"].mean()
#             plt.figure(figsize=(8,5)); sr_by_bucket.plot(kind="bar")
#             plt.ylabel("Mean Success Rate"); plt.title("Success Rate by Complexity Bucket (episodes)")
#             plt.tight_layout(); plt.savefig(os.path.join(out_dir, "sr_by_bucket.png"), dpi=150); plt.close()

#     # -------- 6) Per-Stage SR --------
#     if "stage" in df.columns:
#         st = df.groupby("stage")["success"].mean()
#         plt.figure(figsize=(8,5)); st.plot(kind="bar")
#         plt.ylabel("Mean Success Rate"); plt.title("Success Rate by Stage (episodes)")
#         plt.tight_layout(); plt.savefig(os.path.join(out_dir, "sr_by_stage.png"), dpi=150); plt.close()


# class ComplexityScheduler:
#     """
#     分阶段课程：
#       - 每个阶段持有一组 maps（由 complexity 分位切分）
#       - 每阶段至少跑 min_episodes_per_stage 集
#       - 达到阈值（滑窗或阶段累计）后晋级
#     """

    
#     def __init__(self,
#                  base_map_settings: Dict[str, dict],
#                  n_stages: int = 5,
#                  min_per_stage: int = 5,
#                  # 判定相关
#                  min_episodes_per_stage: int =200 ,   # 每阶段至少训练这么多 episode
#                  threshold: float = 0.70,             # 成功率阈值
#                  window_size: int = 100,              # 滑窗大小
#                  use_window_sr: bool = True,          # True=用滑窗SR；False=用阶段累计SR
#                  # 其它
#                  shuffle_each_stage: bool = True,
#                  seed: int = 0,
#                  size_mode: str = "max"):
#         self.min_episodes_per_stage = int(min_episodes_per_stage)
#         self.threshold = float(threshold)
#         self.window_size = int(window_size)
#         self.use_window_sr = bool(use_window_sr)

#         self._rng = random.Random(seed)
#         df = _compute_complexities_for_settings(base_map_settings, size_mode=size_mode)
#         stages = _build_stages_by_quantile_df(df, n_stages=n_stages, min_per_stage=min_per_stage)

#         self._stage_items = []
#         self._stage_edges = []
#         for st in stages:
#             items = list(st["items"])
#             if shuffle_each_stage:
#                 self._rng.shuffle(items)
#             self._stage_items.append(items)
#             self._stage_edges.append((st["cpx_min"], st["cpx_max"]))

#         self.current_stage = 0
#         self.max_stage = len(self._stage_items) - 1

#         # 轮转采样索引
#         self._idx_in_stage = 0
#         # 统计
#         self._win = deque(maxlen=self.window_size)  # 滑窗
#         self._ep_in_stage = 0                       # 阶段 episode 数
#         self._succ_in_stage = 0                     # 阶段成功 episode 数
    
    

#     # ============ 取图 ============
#     def get_updated_map_settings(self) -> Dict[str, dict]:
#         if self.current_stage > self.max_stage:
#             return {}
#         items = self._stage_items[self.current_stage]
#         if not items:
#             raise RuntimeError(f"Stage {self.current_stage} 没有地图。")
#         item = items[self._idx_in_stage]
#         self._idx_in_stage = (self._idx_in_stage + 1) % len(items)
#         # spec 中保留 complexity，方便训练侧打印
#         return {item["name"]: item["spec"]}

#     # ============ 统计 & 判定 ============
#     def add_episode_result(self, success: int):
#         s = 1 if success else 0
#         self._win.append(s)
#         self._ep_in_stage += 1
#         self._succ_in_stage += s

#     def window_sr(self) -> float:
#         return float(sum(self._win) / len(self._win)) if len(self._win) else 0.0

#     def stage_sr(self) -> float:
#         return float(self._succ_in_stage / max(1, self._ep_in_stage))

#     def should_advance(self) -> bool:
#         """是否满足晋级条件：跑满最少集数 且 成功率达标"""
#         if self._ep_in_stage < self.min_episodes_per_stage:
#             return False
#         sr = self.window_sr() if self.use_window_sr else self.stage_sr()
#         return sr >= self.threshold

#     # ============ 阶段切换 ============
#     def advance(self, pbar=None):
#         if pbar:
#             lo, hi = self._stage_edges[self.current_stage]
#             pbar.write(
#                 f"✅ 通过 Stage {self.current_stage} | "
#                 f"SR(win)={self.window_sr():.2f} / SR(stage)={self.stage_sr():.2f} | "
#                 f"区间[{lo:.4f}, {hi:.4f}] → Stage {self.current_stage + 1}"
#             )
#         self.current_stage += 1
#         self._reset_stage_stats()


#     def repeat_stage(self, pbar=None):
#         if pbar:
#             pbar.write(f"🔁 未达标，重复 Stage {self.current_stage}（已训练 {self._ep_in_stage} ep，SR={self.stage_sr():.2f}）")
#         self._reset_stage_stats()

#     def _reset_stage_stats(self):
#         self._idx_in_stage = 0
#         self._win.clear()
#         self._ep_in_stage = 0
#         self._succ_in_stage = 0

#     def is_done(self) -> bool:
#         return self.current_stage > self.max_stage


# # ================== 训练相关 ==================
# def get_timestamp() -> str:
#     return datetime.now().strftime('%H-%M-%d-%m-%Y')

# def get_normalized_probs(x: Union[List[float], None], size: int) -> np.ndarray:
#     x = [1] * size if x is None else x + [0] * (size - len(x))
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0)

# def train(
#         model: torch.nn.Module,
#         map_settings: Dict[str, dict],
#         map_probs: Union[List[float], None],
#         num_episodes: int = 300,
#         batch_size: int = 32,
#         decay_range: int = 1000,
#         log_dir: str = 'logs',
#         lr: float = 0.001,
#         replay_buffer_size: int = 1000,
#         device: str = 'cuda',
#         scheduler: Optional[ComplexityScheduler] = None,
#         max_episode_seconds: int = 30,
#         run_dir: Optional[str] = None
#     ) -> DDQNAgent:

#     # === 统一确定输出目录 ===
#     if run_dir is None:
#         from datetime import datetime
#         timestamp = datetime.now().strftime('%H-%M-%d-%m-%Y')
#         run_dir = Path(log_dir) / timestamp
#     else:
#         run_dir = Path(run_dir)

#     run_dir.mkdir(parents=True, exist_ok=True)

#     # TensorBoard 写日志到 run_dir
#     writer = SummaryWriter(log_dir=run_dir)

#     training_logs = []



#     # 初始化第一个 env 获取动作空间
#     first_name = next(iter(map_settings))
#     first_env = build_env_from_raw(map_settings[first_name])
#     agent = DDQNAgent(
#         model,
#         first_env.get_action_space(),
#         lr=lr,
#         decay_range=decay_range,
#         device=device,
#         replay_buffer_size=replay_buffer_size,
#     )

#     pbar = tqdm(range(num_episodes), desc='Episodes', dynamic_ncols=True)

#     episode = 0
#     success_count_total = 0
#     stage_success_count = 0
#     stage_episode_count = 0
#     stage_threshold = getattr(scheduler, "threshold", 0.8) if scheduler else 0.8

#     while (scheduler is None) or (scheduler.current_stage <= scheduler.max_stage):
#         if episode >= num_episodes:
#             break

#         # 1) 阶段地图
#         cur_map_cfg = scheduler.get_updated_map_settings() if scheduler else map_settings
#         map_type, cfg = next(iter(cur_map_cfg.items()))
#         env = build_env_from_raw(cfg)

#         cpx_val = cfg.get("complexity", None)  # ★ 直接从 cfg 里取
#         stage_id = scheduler.current_stage if scheduler else '-'
#         if cpx_val is not None:
#             pbar.write(f"🟢 使用地图：{map_type} | Stage {stage_id} | Agents={env.num_agents} | Complexity={cpx_val:.3f}")
#         else:
#             pbar.write(f"🟢 使用地图：{map_type} | Stage {stage_id} | Agents={env.num_agents}")



#         # 2) reset
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

#         # 3) 一集
#         timesteps_per_episode = 50 + 10 * episode
#         episode_start_time = time.time()

#         for t in range(timesteps_per_episode):
#             if time.time() - episode_start_time > max_episode_seconds:
#                 pbar.write(f"⏰ Episode {episode} 超时（>{max_episode_seconds}s），强制终止")
#                 break

#             actions = [ag.act(o) for ag, o in zip(agents, obs)]
#             obs, reward, terminated, truncated, info = env.step(actions)

#             agent_pos = tuple(obs[target_idx]['global_xy'])
#             done = (agent_pos == goal)
#             terminated[target_idx] = done

#             if done:
#                 success_flag = True
#                 scalars['Success'] = 1
#                 scalars['Moving Cost'] = moving_cost(t + 1, opt_path[0], opt_path[-1])
#                 scalars['Detour Percentage'] = detour_percentage(t + 1, len(opt_path) - 1)
#                 break

#             # 经验
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

#         if retrain_count > 0:
#             scalars['Average Loss'] /= retrain_count
#             scalars['Average Epsilon'] /= retrain_count

#         if success_flag:
#             success_count_total += 1
#             stage_success_count += 1
#         scalars['Success'] = 1 if success_flag else 0
#                 # 统计步数
#         steps_this = t + 1 if 't' in locals() else 0
        


#         if scheduler is not None:
#             scheduler.add_episode_result(scalars['Success'])
#             win_sr = scheduler.window_sr()
#             stg_sr = scheduler.stage_sr()
#     # 也顺手写到 TensorBoard
#             writer.add_scalar('SuccessRate/Window', win_sr, episode)
#             writer.add_scalar('SuccessRate/Stage',  stg_sr, episode)
#         else:
#             win_sr = np.nan
#             stg_sr = np.nan


#         # 记录一条 episodio 日志
#         training_logs.append({
#             "episode": episode,
#             "stage": (scheduler.current_stage if scheduler else -1),
#             "map": map_type,
#             "agents": env.num_agents,
#             "complexity": (cpx_val if cpx_val is not None else np.nan),
#             "success": int(scalars['Success']),
#             "sr_window": (win_sr if scheduler else np.nan),
#             "sr_stage":  (stg_sr if scheduler else np.nan),
#             "reward": float(scalars['Reward']),
#             "steps": int(steps_this),
#             "avg_loss": float(scalars['Average Loss']),
#             "avg_epsilon": float(scalars['Average Epsilon']),
#             "moving_cost": float(scalars.get('Moving Cost', np.nan)),
#             "detour_pct": float(scalars.get('Detour Percentage', np.nan)),

#         })

#         stage_episode_count += 1
#         episode += 1
#         pbar.update(1)

#         for name, value in scalars.items():
#             writer.add_scalar(name, value, episode)

#         pbar.set_postfix(
#             Stage=(scheduler.current_stage if scheduler else "-"),
#             SR_total=f"{success_count_total / max(1, episode):.2f}",
#             R=f"{scalars['Reward']:.2f}",
#         )

#         # 5) 课程逻辑
#         # 5) 课程逻辑：达标晋级（跑满最少集数 + 成功率达标）
#         if scheduler is not None:
            
#             scheduler.add_episode_result(scalars['Success'])

#             if scheduler.should_advance():
#                 scheduler.advance(pbar)
#                 # 进入下一阶段就继续循环（会自动抽取下一阶段的更难地图）
#                 if scheduler.is_done():
#                     break
#                 continue
#             else:
#                 # 没达标但还没满最少集数：继续训练本阶段
#                 # 如果你想加“满最少集数但还未达标→强制重置并重复本阶段”，可加：
#                 if scheduler._ep_in_stage >= scheduler.min_episodes_per_stage:
#                     scheduler.repeat_stage(pbar)
#                 # 然后继续该阶段训练


#         #     # 可选：滑窗快速通道
#         # elif scheduler.ready_to_advance():
#         #         scheduler.advance(pbar)
#         #         stage_success_count = 0
#         #         stage_episode_count = 0
#         #         if scheduler.is_done():
#         #             break

#     final_sr = success_count_total / max(1, episode)
#     agent.final_success_rate = float(final_sr)
#     print(f"[train] final_sr(global) = {success_count_total}/{episode} = {agent.final_success_rate:.6f}")
    
#         # === 保存 CSV & 可视化 ===
#     df_train = pd.DataFrame(training_logs)
#     csv_path = run_dir / "episodes.csv"
#     df_train.to_csv(csv_path, index=False, encoding="utf-8-sig")
#     print(f"📝 训练日志已保存：{csv_path}")

#     try:
#         make_training_plots(str(run_dir), df_train, win=50)
#         print(f"📊 训练可视化已保存到：{run_dir}")
#     except Exception as e:
#         print(f"⚠️ 生成可视化失败：{e}")


#     writer.close()
#     return agent

# # ================== 入口 ==================
# if __name__ == '__main__':
#     os.makedirs('logs', exist_ok=True)
#     os.makedirs('models', exist_ok=True)

#     MAP_SETTINGS_PATH = 'C:/Users/MSc_SEIoT_1/MAPF_G2RL-main/g2rl/map_settings_generated.yaml'
#     with open(MAP_SETTINGS_PATH, "r", encoding="utf-8") as f:
#         base_map_settings = yaml.safe_load(f)

#     if isinstance(base_map_settings, list):
#         base_map_settings = { (m.get("name") or f"map_{i}"): m for i, m in enumerate(base_map_settings) }

#     scheduler = ComplexityScheduler(
#         base_map_settings=base_map_settings,
#         n_stages=5,
#         min_per_stage=10,
#         min_episodes_per_stage=100,
#         threshold=0.85,
#         window_size=100,
#         shuffle_each_stage=True,
#         seed=0,
#         size_mode="max",
#     )

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = CRNNModel().to(device)

#     trained_agent = train(
#         model=model,
#         scheduler=scheduler,
#         map_settings=scheduler.get_updated_map_settings(),  # 初始一张；train 内每集会再取
#         map_probs=None,
#         num_episodes=300,
#         batch_size=32,
#         replay_buffer_size=500,
#         decay_range=10_000,
#         log_dir='logs',
#         device=device,
#         run_dir="C:/Users/MSc_SEIoT_1/MAPF_G2RL-main/final_trainig"
#     )

#     torch.save(model.state_dict(), 'models/model3.pt')
#     print('✅ 模型已保存到 models/model3.pt')


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Test a trained Curriculum Learning model on maps sorted by Complexity,
# and evaluate performance by stages.

# Patched upgrades (incl. action calibration):
# 1) Greedy eval: agent.epsilon = 0.0
# 2) Congestion-aware target selection (free-degree>=2 & shortest guidance; fallback shortest/random)
# 3) Micro-assist when stuck (--assist guidance): only when chosen action is noop
# 4) Hard-case adaptive budgets: if density>=0.50 or agents>=12, ensure steps_factor>=10, secs_factor>=2
# 5) NEW: Runtime action encoding calibration — auto-detect noop / up / down / left / right,
#    and use this mapping for both noop check & guidance step (no more guessing)

# Usage:
# python test_by_complexity.py --model_path models/model1.pt --stages 5 --episodes_per_map 8 \
#   --max_episode_steps 2000 --max_episode_seconds 600 --steps_factor 8.0 --secs_factor 2.0 \
#   --max_complexity 2.8071 --teammates astar --assist guidance
# """

# import os
# import sys
# import math
# import time
# import argparse
# import random
# import inspect
# from dataclasses import dataclass
# from typing import Optional, Dict, Any, List, Tuple

# import numpy as np
# import pandas as pd

# # ======= Complexity config (align with your training) =======
# INTERCEPT = 0.848
# WEIGHTS = {
#     'Size': 0.021,
#     'Agents': -0.010,
#     'Density': 0.077,
#     'Density_actual': 0.132,
#     'LDD': 0.027,
#     'BN': -0.128,
#     'MC': 0.039,
#     'DLR': 0.002,
# }

# # ---------- Project root ----------
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# # ---------- Project imports ----------
# from g2rl.environment import G2RLEnv
# from g2rl.agent import DDQNAgent
# from g2rl.network import CRNNModel
# from g2rl.complexity_module import compute_map_complexity
# from pogema import AStarAgent

# try:
#     import torch
# except Exception:
#     torch = None


# # ------------------------- Data structures -------------------------
# @dataclass
# class MapConfig:
#     size: int
#     num_agents: int
#     density: float
#     obs_radius: int = 5
#     max_episode_steps: int = 100
#     seed: Optional[int] = None
#     max_episode_seconds: int = 30  # wall-clock safety


# @dataclass
# class MapRecord:
#     map_id: str
#     size: int
#     agents: int
#     density: float
#     seed: int
#     complexity: float
#     LDD: Optional[float] = None
#     BN: Optional[float] = None
#     MC: Optional[float] = None
#     DLR: Optional[float] = None


# # ------------------------- Complexity helpers -------------------------
# def _parse_complexity_result(comp) -> Tuple[float, float, float, float, float]:
#     import numpy as np
#     if isinstance(comp, dict):
#         return (
#             float(comp.get("Complexity", np.nan)),
#             float(comp.get("LDD", np.nan)),
#             float(comp.get("BN", np.nan)),
#             float(comp.get("MC", np.nan)),
#             float(comp.get("DLR", np.nan)),
#         )
#     if isinstance(comp, (tuple, list)):
#         if len(comp) == 1 and isinstance(comp[0], (int, float)):
#             return float(comp[0]), np.nan, np.nan, np.nan, np.nan
#         if len(comp) == 2 and isinstance(comp[0], (int, float)) and isinstance(comp[1], dict):
#             sub = comp[1]
#             return (
#                 float(comp[0]),
#                 float(sub.get("LDD", np.nan)),
#                 float(sub.get("BN", np.nan)),
#                 float(sub.get("MC", np.nan)),
#                 float(sub.get("DLR", np.nan)),
#             )
#         if len(comp) == 5 and all(isinstance(x, (int, float)) for x in comp):
#             c1, l1, b1, m1, d1 = comp
#             def looks_like(x): return 0.0 <= x <= 1.0
#             if looks_like(c1) and not looks_like(l1):
#                 return float(c1), float(l1), float(b1), float(m1), float(d1)
#             else:
#                 l2, b2, m2, d2, c2 = comp
#                 return float(c2), float(l2), float(b2), float(m2), float(d2)
#     return np.nan, np.nan, np.nan, np.nan, np.nan


# def compute_complexity_safe(grid, size, agents, density) -> Tuple[float, float, float, float, float]:
#     """Prefer compute_map_complexity; fallback to linear proxy to keep sortable."""
#     import numpy as np
#     try:
#         comp = compute_map_complexity(
#             grid, intercept=INTERCEPT, weights=WEIGHTS,
#         )
#         C, LDD, BN, MC, DLR = _parse_complexity_result(comp)
#     except Exception as e:
#         print(f"[ComplexityWarning] compute failed: {e}")
#         C, LDD, BN, MC, DLR = np.nan, np.nan, np.nan, np.nan, np.nan

#     if C is None or (isinstance(C, float) and np.isnan(C)):
#         C = float(INTERCEPT + WEIGHTS["Size"] * size + WEIGHTS["Agents"] * agents + WEIGHTS["Density"] * density)
#     return float(C), LDD, BN, MC, DLR


# # ------------------------- Core env/utils -------------------------
# def _env_num_agents(env) -> int:
#     gc = getattr(env, "grid_config", None)
#     if gc is not None and hasattr(gc, "num_agents"):
#         try:
#             return int(gc.num_agents)
#         except Exception:
#             pass
#     v = getattr(env, "num_agents", None)
#     return int(v) if isinstance(v, int) and v > 0 else 1


# def _infer_num_actions(env) -> int:
#     for name in ["action_space_n", "n_actions", "num_actions"]:
#         v = getattr(env, name, None)
#         if isinstance(v, int) and v > 0:
#             return v
#     asp = getattr(env, "action_space", None)
#     if asp is not None and hasattr(asp, "n"):
#         return int(asp.n)
#     gas = getattr(env, "get_action_space", None)
#     if callable(gas):
#         try:
#             obj = gas()
#             if hasattr(obj, "n"):
#                 return int(obj.n)
#             if isinstance(obj, int) and obj > 0:
#                 return obj
#         except Exception:
#             pass
#     return 5  # fallback: UDLR+Stay


# def _get_action_space(env):
#     asp = getattr(env, "action_space", None)
#     if asp is not None:
#         return asp
#     gas = getattr(env, "get_action_space", None)
#     if callable(gas):
#         try:
#             return gas()
#         except Exception:
#             pass
#     from types import SimpleNamespace
#     return SimpleNamespace(n=_infer_num_actions(env))


# def _to_scalar_per_agent(x, target_idx):
#     if isinstance(x, dict):
#         if target_idx in x:
#             return x[target_idx]
#         for k in ["global", "all", "any", "done", "terminated"]:
#             if k in x:
#                 return x[k]
#         try:
#             return next(iter(x.values()))
#         except Exception:
#             return 0
#     if isinstance(x, (list, tuple, np.ndarray)):
#         if len(x) == 0:
#             return 0
#         if target_idx < len(x):
#             return x[target_idx]
#         try:
#             return bool(np.any(x))
#         except Exception:
#             return x[0]
#     return x


# def _env_step_joint(env, joint_actions, target_idx):
#     out = env.step(joint_actions)
#     if isinstance(out, tuple) and len(out) == 5:
#         obs, reward, terminated, truncated, info = out
#     elif isinstance(out, tuple) and len(out) == 4:
#         obs, reward, done, info = out
#         terminated, truncated = done, False
#     else:
#         obs, reward, terminated, truncated, info = out

#     reward_s     = _to_scalar_per_agent(reward,     target_idx)
#     terminated_s = bool(_to_scalar_per_agent(terminated, target_idx))
#     truncated_s  = bool(_to_scalar_per_agent(truncated,  target_idx))
#     return obs, reward_s, terminated_s, truncated_s, info


# def _extract_grid(env) -> Optional[np.ndarray]:
#     """Try to get 2D occupancy grid (0=free,1=obst); return None if unavailable."""
#     def _is_grid(a):
#         try:
#             arr = np.array(a); return arr.ndim == 2
#         except Exception:
#             return False
#     for name in ["grid", "grid_map", "grid_matrix", "grid_array", "occupancy", "occupancy_grid", "obstacle_map", "map", "matrix"]:
#         a = getattr(env, name, None)
#         if _is_grid(a): arr = np.array(a); return (arr > 0).astype(np.uint8)
#     gc = getattr(env, "grid_config", None)
#     if gc is not None:
#         for name in ["grid", "grid_map", "grid_matrix", "grid_array", "occupancy", "occupancy_grid", "obstacle_map", "map", "matrix"]:
#             a = getattr(gc, name, None)
#             if _is_grid(a): arr = np.array(a); return (arr > 0).astype(np.uint8)
#     return None


# def _get_pos_goal_from_env_and_state(env, state, target_idx):
#     pos = goal = None
#     try:
#         if hasattr(env, "goals"):
#             g = env.goals[target_idx]; goal = (int(g[0]), int(g[1]))
#     except Exception: pass
#     try:
#         if hasattr(env, "starts"):
#             s = env.starts[target_idx]; pos = (int(s[0]), int(s[1]))
#     except Exception: pass
#     if isinstance(state, dict):
#         if pos is None:
#             gx = state.get("global_xy") or state.get("pos")
#             if gx is not None:
#                 try: pos = (int(gx[0]), int(gx[1]))
#                 except Exception: pass
#         if goal is None:
#             gt = state.get("global_target_xy") or state.get("goal")
#             if gt is not None:
#                 try: goal = (int(gt[0]), int(gt[1]))
#                 except Exception: pass
#     return pos, goal


# def _estimate_opt_len(env, state, target_idx, start_pos, goal_pos):
#     try:
#         path = [state["global_xy"]] + env.global_guidance[target_idx]
#         return max(1, len(path) - 1)
#     except Exception:
#         pass
#     if start_pos is not None and goal_pos is not None:
#         manhattan = abs(start_pos[0] - goal_pos[0]) + abs(start_pos[1] - goal_pos[1])
#         return max(1, int(manhattan))
#     return 1


# # ------------------------- Build envs & sampling -------------------------
# def _make_env_from_config(cfg: MapConfig) -> G2RLEnv:
#     env = G2RLEnv(
#         size=cfg.size,
#         num_agents=cfg.num_agents,
#         density=cfg.density,
#         obs_radius=cfg.obs_radius,
#         max_episode_steps=cfg.max_episode_steps,
#         seed=cfg.seed,
#     )
#     if not hasattr(env, "_args"):
#         setattr(env, "_args", {})
#     env._args["max_episode_seconds"] = int(cfg.max_episode_seconds)
#     return env


# def _reset_env_with_seed_and_grid(env: G2RLEnv, seed: int) -> Tuple[Any, Dict]:
#     try:
#         return env.reset(seed=seed)
#     except TypeError:
#         obs = env.reset()
#         return (obs, {})


# def _extract_grid_or_synth(env, size, density, seed):
#     try:
#         grid = _extract_grid(env)
#         if isinstance(grid, np.ndarray) and grid.ndim == 2:
#             return grid
#         raise ValueError("invalid grid")
#     except Exception:
#         rng = np.random.RandomState(seed)
#         return (rng.rand(size, size) < float(density)).astype(np.uint8)


# def sample_grid_maps() -> List[MapRecord]:
#     """Grid combos fixed (CL-consistent) so you have stable staged buckets."""
#     map_sizes    = [32, 64, 96, 128]
#     agent_counts = [2, 4, 8, 16]
#     densities    = [0.10, 0.30, 0.50, 0.70]

#     records: List[MapRecord] = []
#     idx = 0
#     for size in map_sizes:
#         for agents in agent_counts:
#             for density in densities:
#                 seed = 1000 + idx
#                 idx += 1
#                 try:
#                     cfg = MapConfig(size=size, num_agents=agents, density=density, obs_radius=5, max_episode_steps=100, seed=seed, max_episode_seconds=30)
#                     env = _make_env_from_config(cfg)
#                 except Exception as e:
#                     print(f"[ENV-ERROR] build env failed s={size} a={agents} d={density:.2f}: {e}")
#                     grid = (np.random.RandomState(seed).rand(size, size) < density).astype(np.uint8)
#                     C,LDD,BN,MC,DLR = compute_complexity_safe(grid, size, agents, density)
#                     records.append(MapRecord(f"grid_{idx}", size, agents, density, seed, C, LDD, BN, MC, DLR))
#                     continue

#                 try:
#                     _reset_env_with_seed_and_grid(env, seed)
#                 except Exception as e:
#                     print(f"[ENV-WARN] reset(seed) failed: {e}")
#                     try: env.reset()
#                     except Exception as e2: print(f"[ENV-ERROR] reset() failed: {e2}")

#                 grid = _extract_grid_or_synth(env, size, density, seed)
#                 C, LDD, BN, MC, DLR = compute_complexity_safe(grid, size, agents, density)

#                 records.append(MapRecord(
#                     map_id=f"grid_{idx}", size=size, agents=agents, density=density,
#                     seed=seed, complexity=float(C), LDD=LDD, BN=BN, MC=MC, DLR=DLR
#                 ))

#     return records


# def split_into_stages(records: List[MapRecord], stages: int, max_complexity: Optional[float]) -> List[List[MapRecord]]:
#     # filter by max_complexity if given
#     if max_complexity is not None:
#         records = [r for r in records if (r.complexity is not None and not np.isnan(r.complexity) and r.complexity <= max_complexity)]
#     # sort by complexity (NaN removed already)
#     records = sorted(records, key=lambda r: r.complexity)
#     chunk = max(1, math.ceil(len(records) / stages))
#     return [records[i:i+chunk] for i in range(0, len(records), chunk)]


# # ------------------------- Patched target selection -------------------------
# def _free_degree(grid_arr, xy):
#     if grid_arr is None or xy is None:
#         return 4
#     H, W = grid_arr.shape
#     x, y = int(xy[0]), int(xy[1])
#     deg = 0
#     for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
#         nx, ny = x+dx, y+dy
#         if 0 <= nx < H and 0 <= ny < W and grid_arr[nx, ny] == 0:
#             deg += 1
#     return deg


# def _choose_target_idx(env, obs) -> int:
#     """Congestion-aware: prefer guidance-short & start free-degree>=2."""
#     num_agents = _env_num_agents(env)
#     gl = getattr(env, "global_guidance", None)
#     grid_arr = None
#     try:
#         grid_arr = _extract_grid(env)
#     except Exception:
#         pass

#     if gl is not None:
#         cands: List[Tuple[int,int]] = []
#         for i in range(num_agents):
#             try:
#                 L = len(gl[i])
#                 s_i, _ = _get_pos_goal_from_env_and_state(env, obs[i] if isinstance(obs, (list, tuple)) else obs, i)
#                 if L > 0 and _free_degree(grid_arr, s_i) >= 2:
#                     cands.append((L, i))
#             except Exception:
#                 pass
#         if cands:
#             cands.sort()
#             return cands[0][1]
#         # fallback: shortest guidance
#         best = None
#         for i in range(num_agents):
#             try:
#                 L = len(gl[i])
#                 if L > 0 and (best is None or L < best[0]):
#                     best = (L, i)
#             except Exception:
#                 pass
#         if best is not None:
#             return best[1]

#     # ultimate fallback: random
#     return np.random.randint(num_agents)


# # ------------------------- Action encoding calibration -------------------------
# _CALIB_CACHE = {}

# def calibrate_action_encoding(env, target_idx: int):
#     """
#     Auto-detect mapping: noop, up, down, left, right.
#     Do one-step probing per action; cache per-env instance.
#     """
#     key = id(env)
#     if key in _CALIB_CACHE:
#         return _CALIB_CACHE[key]

#     # reset & current pos
#     try:
#         obs, _ = env.reset(seed=getattr(env, "seed", None))
#     except TypeError:
#         obs = env.reset()

#     num_agents = _env_num_agents(env)
#     n_actions = _infer_num_actions(env)

#     def _state_of(o, ti):
#         try: return o[ti]
#         except Exception: return o

#     state = _state_of(obs, target_idx)
#     cur_xy, _ = _get_pos_goal_from_env_and_state(env, state, target_idx)

#     effects: Dict[int, Tuple[int,int]] = {}
#     for a in range(n_actions):
#         # isolate step per action
#         try:
#             env.reset(seed=getattr(env, "seed", None))
#         except TypeError:
#             env.reset()
#         joint = [0] * num_agents   # assume 0 is safe placeholder (usually stay)
#         joint[target_idx] = a
#         out = env.step(joint)
#         obs2 = out[0] if isinstance(out, tuple) else out
#         nxt_state = _state_of(obs2, target_idx)
#         nxt_xy, _ = _get_pos_goal_from_env_and_state(env, nxt_state, target_idx)
#         if cur_xy is None or nxt_xy is None:
#             dv = (0, 0)
#         else:
#             dv = (int(nxt_xy[0] - cur_xy[0]), int(nxt_xy[1] - cur_xy[1]))  # (dx,dy) in env coords
#         effects[a] = dv

#     # noop = smallest motion
#     noop = min(effects.keys(), key=lambda k: abs(effects[k][0]) + abs(effects[k][1]))

#     # try exact match for common deltas
#     dir_map = {}
#     exact = {(-1,0): "up", (1,0): "down", (0,-1): "left", (0,1): "right"}
#     used = {noop}
#     for a, dv in effects.items():
#         if a == noop:
#             continue
#         if dv in exact and exact[dv] not in dir_map:
#             dir_map[exact[dv]] = a
#             used.add(a)

#     # fill missing by nearest
#     def _closest(target):
#         best = None
#         for a, dv in effects.items():
#             if a in used:
#                 continue
#             dist = (dv[0]-target[0])**2 + (dv[1]-target[1])**2
#             if best is None or dist < best[0]:
#                 best = (dist, a)
#         return best[1] if best else noop

#     for name, vec in {"up":(-1,0), "down":(1,0), "left":(0,-1), "right":(0,1)}.items():
#         if name not in dir_map:
#             dir_map[name] = _closest(vec)
#             used.add(dir_map[name])

#     mapping = {"noop": noop, **dir_map}
#     _CALIB_CACHE[key] = mapping
#     print(f"[Calibrate] mapping={mapping} effects={effects}")
#     return mapping


# def toward_with_mapping(dx: int, dy: int, mapping: Dict[str, int]) -> int:
#     """
#     Choose the mapped action whose direction best aligns with (dx,dy).
#     """
#     # simple nearest by negative dot (align more -> smaller)
#     best = None
#     for name, vec in {"up":(-1,0), "down":(1,0), "left":(0,-1), "right":(0,1)}.items():
#         score = -(dx*vec[0] + dy*vec[1])
#         if best is None or score < best[0]:
#             best = (score, name)
#     return mapping.get(best[1], mapping["noop"])


# # ------------------------- Agent loader (epsilon=0) -------------------------
# def _load_agent(model_path: str, env: G2RLEnv) -> DDQNAgent:
#     num_actions = _infer_num_actions(env)
#     net = CRNNModel(num_actions=num_actions)
#     action_space = _get_action_space(env)
#     agent = DDQNAgent(net, action_space)
#     if torch is not None and os.path.exists(model_path):
#         try:
#             ckpt = torch.load(model_path, map_location="cpu", weights_only=True)  # PyTorch≥2.4
#         except TypeError:
#             ckpt = torch.load(model_path, map_location="cpu")
#         state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
#         try:
#             net.load_state_dict(state, strict=False)
#         except Exception as e:
#             print("[LoadWarning] net.load_state_dict failed:", e)
#     agent.eval_mode = True
#     if hasattr(agent, "epsilon"):
#         agent.epsilon = 0.0           # patch 1: greedy eval
#     return agent


# # ------------------------- Episode rollout (patched + calibrated) -------------------------
# def _rollout_one_episode(env, agent, target_idx: Optional[int] = None, teammates: str = "astar",
#                          steps_factor: float = 5.0, secs_factor: float = 1.0,
#                          assist: str = "none") -> Dict[str, Any]:
#     """Single-episode eval with patches & action calibration."""
#     try:
#         obs, info = env.reset(seed=getattr(env, "seed", None))
#     except TypeError:
#         obs = env.reset()
#         info = {}

#     num_agents = _env_num_agents(env)

#     # pick target (patched): congestion-aware
#     if target_idx is None:
#         target_idx = _choose_target_idx(env, obs)

#     # actions (calibrate!)
#     mapping = calibrate_action_encoding(env, target_idx)
#     noop = mapping["noop"]
#     n_actions = _infer_num_actions(env)

#     def _state_of(o, ti):
#         try:
#             return o[ti]
#         except Exception:
#             return o

#     state = _state_of(obs, target_idx)
#     start_pos, goal_pos = _get_pos_goal_from_env_and_state(env, state, target_idx)
#     last_pos = start_pos

#     # opt & budgets
#     opt_len = _estimate_opt_len(env, state, target_idx, start_pos, goal_pos)
#     base_steps = int(getattr(env, "max_episode_steps", None)
#                      or getattr(getattr(env, "grid_config", env), "max_episode_steps", None)
#                      or (getattr(env, "size", 32) * 3))
#     base_timeout = int(getattr(env, "_args", {}).get("max_episode_seconds", 30))
#     max_steps = max(base_steps, int(opt_len * steps_factor))
#     timeout_s = max(base_timeout, int(opt_len * secs_factor))

#     # patch 4: hard-case uplift
#     density = getattr(env, "density", getattr(getattr(env, "grid_config", object), "density", 0.0))
#     agents  = getattr(env, "num_agents", getattr(getattr(env, "grid_config", object), "num_agents", 1))
#     if (float(density) >= 0.50) or (int(agents) >= 12):
#         max_steps = max(max_steps, int(opt_len * max(steps_factor, 10.0)))
#         timeout_s = max(timeout_s, int(opt_len * max(secs_factor,  2.0)))

#     # teammates
#     astar_team = None
#     if teammates == "astar":
#         astar_team = [AStarAgent() if i != target_idx else None for i in range(num_agents)]

#     t0 = time.time()
#     total_reward = 0.0
#     steps = 0
#     success_flag = 0
#     non_noop_moves = 0
#     path_len = 0

#     def _sanitize_action(a, n):
#         try:
#             ai = int(a)
#         except Exception:
#             ai = 0
#         if n > 0:
#             ai = ai % n
#         return ai

#     for _ in range(max_steps):
#         steps += 1
#         if time.time() - t0 > timeout_s:
#             break

#         joint = [noop] * num_agents

#         # teammates act
#         if astar_team is not None:
#             for i in range(num_agents):
#                 if i == target_idx:
#                     continue
#                 try:
#                     joint[i] = int(astar_team[i].act(obs[i]))
#                 except Exception:
#                     joint[i] = noop

#         # our agent act (greedy)
#         try:
#             a = agent.select_action(state, eval_mode=True)
#             act = int(a) if np.isscalar(a) else int(np.argmax(np.array(a)))
#         except Exception:
#             act = random.randrange(n_actions)
#         joint[target_idx] = act

#         # micro-assist when noop (use calibrated mapping)
#         if str(assist).lower() == "guidance" and joint[target_idx] == noop:
#             try:
#                 gpath = getattr(env, "global_guidance", None)
#                 if gpath is not None and len(gpath[target_idx]) > 0:
#                     cur_xy, _ = _get_pos_goal_from_env_and_state(env, state, target_idx)
#                     if cur_xy is not None:
#                         nxt = gpath[target_idx][0]
#                         dx, dy = int(nxt[0]-cur_xy[0]), int(nxt[1]-cur_xy[1])
#                         joint[target_idx] = toward_with_mapping(dx, dy, mapping)
#             except Exception:
#                 pass

#         # sanitize / align length
#         joint = [_sanitize_action(a, n_actions) for a in joint]
#         if len(joint) < num_agents:
#             joint += [noop] * (num_agents - len(joint))
#         elif len(joint) > num_agents:
#             joint = joint[:num_agents]

#         obs_next, reward, terminated, truncated, info = _env_step_joint(env, joint, target_idx)
#         try:
#             total_reward += float(reward)
#         except Exception:
#             total_reward += float(np.mean(reward))

#         if joint[target_idx] != noop:
#             non_noop_moves += 1
#             path_len += 1

#         next_state = _state_of(obs_next, target_idx)
#         obs = obs_next

#         pos_now, _ = _get_pos_goal_from_env_and_state(env, next_state, target_idx)
#         if pos_now is not None:
#             last_pos = pos_now

#         # success by reaching goal
#         if goal_pos is not None and last_pos is not None and last_pos == goal_pos:
#             success_flag = 1
#             state = next_state
#             break

#         # env termination
#         if terminated or truncated:
#             if isinstance(info, dict):
#                 info_ai = info.get(target_idx, info) if isinstance(info.get(target_idx, None), dict) else info
#                 if isinstance(info_ai, dict):
#                     for k in ("success", "is_success", "solved", "reached_goal", "done"):
#                         if k in info_ai:
#                             try:
#                                 success_flag = int(bool(info_ai[k]))
#                                 break
#                             except Exception:
#                                 pass
#             if success_flag == 0 and goal_pos is not None and last_pos is not None:
#                 success_flag = int(last_pos == goal_pos)
#             state = next_state
#             break

#         state = next_state

#     if opt_len <= 0:
#         opt_len = 1
#     detour_pct = max(0.0, (path_len - opt_len) / opt_len * 100.0)

#     return {
#         "steps": steps,
#         "reward": float(total_reward),
#         "success": int(success_flag),
#         "moving_cost": float(non_noop_moves),
#         "detour_pct": float(detour_pct),
#         "opt_len": float(opt_len),
#     }


# # ------------------------- Stage evaluation -------------------------
# def evaluate_agent_on_maps(args, stage_id: int, maps: List[MapRecord]) -> pd.DataFrame:
#     rows = []
#     agent_global = None

#     for rec in maps:
#         cfg = MapConfig(
#             size=rec.size,
#             num_agents=rec.agents,
#             density=rec.density,
#             obs_radius=args.obs_radius,
#             max_episode_steps=args.max_episode_steps,
#             seed=rec.seed,
#             max_episode_seconds=args.max_episode_seconds,
#         )
#         env = _make_env_from_config(cfg)
#         _reset_env_with_seed_and_grid(env, rec.seed)

#         if agent_global is None:
#             agent_global = _load_agent(args.model_path, env)
#         agent = agent_global

#         ep_metrics = []
#         for _ in range(args.episodes_per_map):
#             m = _rollout_one_episode(
#                 env, agent,
#                 teammates=args.teammates,
#                 steps_factor=args.steps_factor,
#                 secs_factor=args.secs_factor,
#                 assist=args.assist
#             )
#             ep_metrics.append(m)

#         # aggregate
#         success_list = [m['success'] for m in ep_metrics]
#         row = {
#             'stage': stage_id,
#             'map': rec.map_id,
#             'size': rec.size,
#             'agents': rec.agents,
#             'density': rec.density,
#             'seed': rec.seed,
#             'complexity': rec.complexity,
#             'LDD': rec.LDD,
#             'BN': rec.BN,
#             'MC': rec.MC,
#             'DLR': rec.DLR,
#             'episodes': args.episodes_per_map,
#             'success_rate': float(np.mean(success_list)) if ep_metrics else 0.0,
#             'success_at_k': 1.0 if any(success_list) else 0.0,
#             'avg_reward': float(np.mean([m['reward'] for m in ep_metrics])) if ep_metrics else 0.0,
#             'avg_steps': float(np.mean([m['steps'] for m in ep_metrics])) if ep_metrics else 0.0,
#             'avg_moving_cost': float(np.mean([m['moving_cost'] for m in ep_metrics])) if ep_metrics else 0.0,
#             'avg_detour_pct': float(np.mean([m['detour_pct'] for m in ep_metrics])) if ep_metrics else 0.0,
#             'avg_opt_len': float(np.mean([m['opt_len'] for m in ep_metrics])) if ep_metrics else 0.0,
#         }
#         rows.append(row)

#     return pd.DataFrame(rows)


# # ------------------------- Main -------------------------
# def main():
#     parser = argparse.ArgumentParser(description="Test trained CL model by map complexity stages (patched + calibrated)")
#     parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint (.pt)')
#     parser.add_argument('--out_csv', type=str, default='test_by_complexity_results.csv')
#     parser.add_argument('--num_maps', type=int, default=40)
#     parser.add_argument('--stages', type=int, default=5)
#     parser.add_argument('--episodes_per_map', type=int, default=8)

#     parser.add_argument('--size_min', type=int, default=16)
#     parser.add_argument('--size_max', type=int, default=64)
#     parser.add_argument('--agents_min', type=int, default=2)
#     parser.add_argument('--agents_max', type=int, default=16)
#     parser.add_argument('--density_min', type=float, default=0.05)
#     parser.add_argument('--density_max', type=float, default=0.45)

#     parser.add_argument('--obs_radius', type=int, default=5)
#     parser.add_argument('--max_episode_steps', type=int, default=1200)
#     parser.add_argument('--max_episode_seconds', type=int, default=300)

#     parser.add_argument('--steps_factor', type=float, default=5.0)
#     parser.add_argument('--secs_factor',  type=float, default=1.0)

#     parser.add_argument('--teammates', type=str, default='astar', choices=['astar', 'noop'],
#                         help='Teammates: astar (default) or noop')
#     parser.add_argument('--assist', type=str, default='none', choices=['none','guidance'],
#                         help='Micro-assist: guidance (only when noop)')

#     parser.add_argument('--max_complexity', type=float, default=None,
#                         help='If set, filter maps with complexity <= this value')

#     args = parser.parse_args()

#     # 1) sample & compute complexity
#     records = sample_grid_maps()
#     valid = [r for r in records if r.complexity is not None and not np.isnan(r.complexity)]
#     print(f"[INFO] complexity valid={len(valid)} invalid={len(records)-len(valid)}")
#     if not valid:
#         raise RuntimeError("No valid maps with computed complexity.")

#     # 2) split into stages with max_complexity filter
#     stage_buckets = split_into_stages(valid, args.stages, args.max_complexity)

#     # 3) evaluate
#     all_results: List[pd.DataFrame] = []
#     for s, maps in enumerate(stage_buckets):
#         if not maps:
#             continue
#         lo = maps[0].complexity if maps[0].complexity is not None and not np.isnan(maps[0].complexity) else float('nan')
#         hi = maps[-1].complexity if maps[-1].complexity is not None and not np.isnan(maps[-1].complexity) else float('nan')
#         print(f"▶️ Testing Stage {s} - {len(maps)} maps (complexity {lo:.4f} .. {hi:.4f})")
#         df_stage = evaluate_agent_on_maps(args, s, maps)
#         all_results.append(df_stage)

#     if not all_results:
#         print("No results produced.")
#         return

#     results = pd.concat(all_results, ignore_index=True)

#     # 4) save details
#     os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
#     results.to_csv(args.out_csv, index=False, encoding='utf-8-sig')
#     print(f"✅ Saved: {args.out_csv}")

#     # 5) per-stage summary
#     summary = results.groupby('stage').agg({
#         'success_rate': 'mean',
#         'success_at_k': 'mean',
#         'avg_reward': 'mean',
#         'avg_steps': 'mean',
#         'avg_moving_cost': 'mean',
#         'avg_detour_pct': 'mean',
#         'complexity': ['min', 'max', 'mean']
#     })
#     summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
#     print('\n===== Stage Summary =====')
#     print(summary.to_string())

#     base, ext = os.path.splitext(args.out_csv)
#     summary_path = base + '_stage_summary.csv'
#     summary.to_csv(summary_path, encoding='utf-8-sig')
#     print(f"✅ Saved: {summary_path}")


# if __name__ == '__main__':
#     main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union, List
from collections import deque
import os, sys, time, inspect, random

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm
from pogema import AStarAgent

# ============ 项目根路径 ============
project_root = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main-nn"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============ 项目模块 ============
from g2rl.environment import G2RLEnv
from g2rl.agent import DDQNAgent
from g2rl.network import CRNNModel
from g2rl.complexity import compute_complexity

# ============ 安全构造器：只传 __init__ 支持的参数 ============
def build_env_from_raw(raw_cfg: dict) -> G2RLEnv:
    sig = inspect.signature(G2RLEnv.__init__)
    allowed = {
        p.name for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    allowed.discard("self")

    rename = {
        # 如需要可在此做键名映射: 例如 'size': 'map_size'
    }

    ctor_cfg = {}
    for k, v in raw_cfg.items():
        kk = rename.get(k, k)
        if kk in allowed:
            ctor_cfg[kk] = v

    env = G2RLEnv(**ctor_cfg)

    # 将 grid/starts/goals 等作为属性挂载
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
FEATURE_MEAN_STD = None

def _compute_complexities_for_settings(base_map_settings: Dict[str, dict],
                                       size_mode: str = "max") -> pd.DataFrame:
    rows = []
    for name, spec in base_map_settings.items():
        try:
            cpx, used, raw = compute_complexity(
                spec, intercept=INTERCEPT, weights=WEIGHTS,
                feature_mean_std=FEATURE_MEAN_STD, size_mode=size_mode
            )
            spec_with_cpx = dict(spec)
            spec_with_cpx["complexity"] = float(cpx)
            rows.append({"name": name, "complexity": float(cpx), "spec": spec_with_cpx})
        except Exception as e:
            rows.append({"name": name, "error": str(e), "spec": spec})
    df = pd.DataFrame(rows)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    return df.sort_values("complexity").reset_index(drop=True)

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
            "items": sub.to_dict("records"),
        })
    return stages

class ComplexityScheduler:
    """
    - 每阶段持有一组 maps（由 complexity 分位切分）
    - 每阶段至少跑 min_episodes_per_stage 集
    - 满足阈值后晋级
    """
    def __init__(self,
                 base_map_settings: Dict[str, dict],
                 n_stages: int = 5,
                 min_per_stage: int = 10,
                 min_episodes_per_stage: int = 100,
                 threshold: float = 0.70,
                 window_size: int = 100,
                 use_window_sr: bool = True,
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

        self._idx_in_stage = 0
        self._win = deque(maxlen=self.window_size)
        self._ep_in_stage = 0
        self._succ_in_stage = 0

    def get_updated_map_settings(self) -> Dict[str, dict]:
        if self.current_stage > self.max_stage:
            return {}
        items = self._stage_items[self.current_stage]
        if not items:
            raise RuntimeError(f"Stage {self.current_stage} 没有地图。")
        item = items[self._idx_in_stage]
        self._idx_in_stage = (self._idx_in_stage + 1) % len(items)
        return {item["name"]: item["spec"]}

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
        if self._ep_in_stage < self.min_episodes_per_stage:
            return False
        sr = self.window_sr() if self.use_window_sr else self.stage_sr()
        return sr >= self.threshold

    def advance(self, pbar=None):
        
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

# ============ 训练 ============
def get_timestamp() -> str:
    return datetime.now().strftime('%H-%M-%d-%m-%Y')

def train(
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
    scheduler: Optional[ComplexityScheduler] = None,
    max_episode_seconds: int = 30,
    run_dir: Optional[str] = None
) -> DDQNAgent:

    timestamp = get_timestamp()
    run_dir = Path(log_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    # 初始化第一个 env 以获取动作空间
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

    training_logs = []
    pbar = tqdm(range(num_episodes), desc='Episodes', dynamic_ncols=True)

    episode = 0
    success_count_total = 0
    stage_success_count = 0
    stage_episode_count = 0

    while (scheduler is None) or (scheduler.current_stage <= scheduler.max_stage):
        if episode >= num_episodes:
            break

        # 1) 获取当前阶段地图并构建 env
        cur_map_cfg = scheduler.get_updated_map_settings() if scheduler else map_settings
        map_type, cfg = next(iter(cur_map_cfg.items()))
        env = build_env_from_raw(cfg)

        stage_id = (scheduler.current_stage if scheduler else -1)
        cpx_val = cfg.get("complexity", None)
        if cpx_val is not None:
            pbar.write(f"🟢 使用地图：{map_type} | Stage {stage_id} | Agents={env.num_agents} | Complexity={cpx_val:.3f}")
        else:
            pbar.write(f"🟢 使用地图：{map_type} | Stage {stage_id} | Agents={env.num_agents}")

        # 2) reset
        try:
            obs, info = env.reset()
        except Exception:
            obs = env.reset()
            info = {}



        target_idx = np.random.randint(env.num_agents)
        agents = [agent if i == target_idx else AStarAgent() for i in range(env.num_agents)]
        goal = tuple(env.goals[target_idx])
        state = obs[target_idx]

        # 3) 跑一集
        success_flag = False
        retrain_count = 0
        timesteps_per_episode = 50 + 10 * episode
        episode_start_time = time.time()

        for t in range(timesteps_per_episode):
            if time.time() - episode_start_time > max_episode_seconds:
                pbar.write(f"⏰ Episode {episode} 超时（>{max_episode_seconds}s），终止本集")
                break

            actions = [ag.act(o) for ag, o in zip(agents, obs)]
            obs, reward, terminated, truncated, info = env.step(actions)

            # 到达判定
            agent_pos = tuple(obs[target_idx]['global_xy'])
            done = (agent_pos == goal)
            terminated[target_idx] = done

            if done:
                success_flag = True
                break

            # 经验 & 学习
            agent.store(
                state,
                actions[target_idx],
                reward[target_idx],
                obs[target_idx],
                terminated[target_idx],
            )
            state = obs[target_idx]

            if len(agent.replay_buffer) >= batch_size:
                retrain_count += 1
                _ = agent.retrain(batch_size)

        # 4) 更新计数/成功率（小数 0~1）
        if success_flag:
            success_count_total += 1
            stage_success_count += 1

        stage_episode_count += 1
        episode += 1
        success_rate = success_count_total / max(1, episode)   # ← 全局累计成功率（小数）
        writer.add_scalar('success_rate', success_rate, episode)
        writer.add_scalar('success', 1 if success_flag else 0, episode)

        # 进度条展示
        pbar.set_postfix(
            Stage=(stage_id if scheduler else "-"),
            success=int(success_flag),
            success_rate=f"{success_rate:.3f}",
        )
        pbar.update(0)  # 已在 for 外层推进

        # 保存逐集日志（CSV）
        training_logs.append({
            "episode": episode,
            "stage": stage_id,
            "map": map_type,
            "agents": getattr(env, "num_agents", None),
            "complexity": (float(cpx_val) if cpx_val is not None else np.nan),
            "success": int(success_flag),
            "success_rate": float(success_rate),
        })

        # 5) 课程逻辑：把结果写入 scheduler，并根据阈值决定晋级/重复
        if scheduler is not None:
            scheduler.add_episode_result(int(success_flag))
            if scheduler.should_advance():
                scheduler.advance(pbar)
                if scheduler.is_done():
                    break
            else:
                # 如果已达最少集数但未达标 → 重复阶段
                if scheduler._ep_in_stage >= scheduler.min_episodes_per_stage:
                    scheduler.repeat_stage(pbar)

    # 收尾：保存 CSV、关闭 writer
    df = pd.DataFrame(training_logs)
    csv_path = run_dir / "episodes.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"📝 每集日志已保存：{csv_path}")

    writer.close()
    return agent

# ============ 主程序 ============
if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    MAP_SETTINGS_PATH = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main - train/g2rl/map_settings_generated_new.yaml"
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
        use_window_sr=True,
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
        max_episode_seconds=30,
        run_dir="C:/Users/MSc_SEIoT_1/MAPF_G2RL-main/final_trainig_1"
    )

    # 保存权重
    out_path = Path("models") / "best_model.pt"
    torch.save(model.state_dict(), out_path.as_posix())
    print(f"✅ 模型已保存到 {out_path}")


