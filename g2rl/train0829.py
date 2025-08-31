
# """
# train0829.py  —  分层课程(细化) + 风险感知 + 拥堵/等待塑形 + 冷却/拉黑 + 兜底避坑
# 兼容此前用户工程：G2RLEnv / DDQNAgent / CRNNModel

# 要点改动：
# 1) 课程细化：将原 Stage 2 按复杂度一分为二 => Stage 2a(更易) / Stage 2b(较难)
# 2) Stage 2a 强控 agents≤4；Stage 2b 控制 agents≤8；并加重等待/避撞奖励
# 3) 风险感知动作选择：预判下一步冲突/边交换，动作打折或屏蔽
# 4) 拥堵/等待塑形：局部拥堵罚、主动等待奖、时间罚更轻
# 5) 经验回放混采：Stage2 系列采用 成功60% / 近期35% / 困难5%
# 6) 地图调度：冷却/拉黑/避免 cpx≈0.102，候选全冷却时兜底轮换“最易前K”
# 7) tqdm __bool__异常修复：严禁对 pbar 作 bool 判断，统一用 if pbar is not None
# """

# from pathlib import Path
# from datetime import datetime
# from typing import Dict, Optional, Union, List
# from types import SimpleNamespace
# import os, sys, time, inspect, random
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.tensorboard import SummaryWriter
# import yaml
# from tqdm import tqdm
# from collections import deque, defaultdict

# # ===================== 配置 =====================
# # 1) YAML 地图配置
# MAP_SETTINGS_PATH = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main - train/g2rl/map_settings_generated_new.yaml"

# # 2) 复杂度 CSV（由 infer_complexity.py 生成）
# COMPLEXITY_CSV = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main-nn/0827result/maps_features_with_complexity.csv"

# # 3) 训练超参（全局默认）
# N_STAGES = 6                          # 0,1,2a,2b,3,4
# MIN_PER_STAGE = 8
# MIN_EPISODES_PER_STAGE = 300
# THRESHOLD = 0.80
# WINDOW_SIZE = 120
# BATCH_SIZE = 128
# REPLAY_BUFFER_SIZE = 32000
# DECAY_RANGE = 10_000
# MAX_EPISODE_SECONDS = 70
# LOG_DIR = "logs"
# RUN_DIR = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main-nn/final_training_refined"
# MODEL_OUT = "models/best_model.pt"

# # 防卡死：目标智能体连续多少步位置不变就提前终止该集（0=关闭）
# STUCK_PATIENCE = 0
# # ============================================================

# # ============ 项目根路径（确保能 import g2rl.*） ============
# project_root = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main-nn"
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# # ============ 项目模块 ============
# from g2rl.environment import G2RLEnv
# from g2rl.agent import DDQNAgent
# from g2rl.network import CRNNModel

# # ============ 可选：Pogema A* ============
# try:
#     from pogema import AStarAgent as _PogemaAStarAgent
#     HAVE_POGEMA = True
# except Exception:
#     HAVE_POGEMA = False

# # ================= 公用工具 =================
# def _pwrite(pbar, msg: str):
#     """安全写进度条；pbar可能没有total，或为None。"""
#     if pbar is not None:
#         try:
#             pbar.write(msg); return
#         except Exception:
#             pass
#     print(msg)

# class GreedyAgent:
#     """简单贪心：尽量缩小到目标的曼哈顿距离；不明确动作语义时退回 idle。"""
#     def __init__(self, env, idle_idx: int, idx_to_move: Dict[int, tuple]):
#         self.env = env
#         self.idle_idx = idle_idx
#         self.idx_to_move = idx_to_move

#     def act(self, obs_i, goal_xy):
#         try:
#             x, y = obs_i['global_xy']; gx, gy = goal_xy
#         except Exception:
#             return self.idle_idx

#         best_a, best_gain = self.idle_idx, 0
#         d0 = abs(x - gx) + abs(y - gy)
#         grid = getattr(self.env, "grid", None)
#         H, W = (grid.shape[:2] if grid is not None else (None, None))

#         for a, (dx, dy) in self.idx_to_move.items():
#             nx, ny = x + dx, y + dy
#             if H is not None and (nx < 0 or ny < 0 or nx >= H or ny >= W):
#                 continue
#             if grid is not None:
#                 try:
#                     if grid[nx, ny] != 0:
#                         continue
#                 except Exception:
#                     pass
#             d1 = abs(nx - gx) + abs(ny - gy)
#             gain = d0 - d1
#             if gain > best_gain:
#                 best_gain, best_a = gain, a
#         return best_a

# def _find_array(obj):
#     import numpy as np
#     try:
#         import torch
#         is_tensor = True
#     except Exception:
#         is_tensor = False

#     if isinstance(obj, np.ndarray):
#         return obj
#     if is_tensor and hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "numpy"):
#         try:
#             return obj.detach().cpu().numpy()
#         except Exception:
#             pass
#     if isinstance(obj, (list, tuple)):
#         for v in obj:
#             arr = _find_array(v)
#             if arr is not None:
#                 return arr
#         return None
#     if isinstance(obj, dict):
#         preferred = ("view_cache","obs","observation","state","tensor","grid","image","local_obs","global_obs")
#         for k in preferred:
#             if k in obj:
#                 arr = _find_array(obj[k])
#                 if arr is not None: return arr
#         for v in obj.values():
#             arr = _find_array(v)
#             if arr is not None: return arr
#         return None
#     try:
#         arr = np.array(obj)
#         if arr.ndim > 0: return arr
#     except Exception:
#         pass
#     return None

# def _obs_to_tensor_CDHW(s, device, expected_c: int):
#     import numpy as np, torch
#     arr = _find_array(s)
#     if arr is None:
#         raise ValueError("无法从观测中提取数组/张量。")
#     arr = np.array(arr)

#     # 归一到 [C,D,H,W]
#     if arr.ndim == 5:
#         if arr.shape[1] == expected_c: arr = arr[0]
#         elif arr.shape[-1] == expected_c: arr = np.transpose(arr, (0,4,1,2,3))[0]
#         else: arr = arr[0]
#     elif arr.ndim == 4:
#         if arr.shape[0] == expected_c: pass
#         elif arr.shape[-1] == expected_c: arr = np.transpose(arr, (3,0,1,2))
#         elif arr.shape[1] == expected_c: arr = np.transpose(arr, (1,0,2,3))
#     elif arr.ndim == 3:
#         arr = arr[None, ...]
#     elif arr.ndim == 2:
#         arr = arr[None, None, ...]
#     else:
#         raise ValueError(f"观测维度不支持：shape={arr.shape}, ndim={arr.ndim}")

#     C,D,H,W = arr.shape
#     if C == expected_c:
#         arr_fixed = arr
#     elif C == 1 and expected_c > 1:
#         arr_fixed = np.repeat(arr, expected_c, axis=0)
#     elif C < expected_c:
#         pad = np.zeros((expected_c - C, D, H, W), dtype=arr.dtype)
#         arr_fixed = np.concatenate([arr, pad], axis=0)
#     else:
#         arr_fixed = arr[:expected_c]
#     return torch.tensor(arr_fixed[None, ...], dtype=torch.float32, device=device)

# def _safe_get_action_space(env):
#     asp = getattr(env, "action_space", None)
#     if asp is not None and hasattr(asp, "n"): return asp
#     gas = getattr(env, "get_action_space", None)
#     if callable(gas):
#         try:
#             out = gas()
#             if hasattr(out, "n"): return out
#             if isinstance(out, (list, tuple)): return SimpleNamespace(n=len(out))
#             if isinstance(out, int) and out > 0: return SimpleNamespace(n=out)
#         except Exception:
#             pass
#     if hasattr(env, "actions"):
#         try:
#             return SimpleNamespace(n=len(env.actions))
#         except Exception:
#             pass
#     try: env.reset()
#     except Exception: pass
#     asp = getattr(env, "action_space", None)
#     if asp is not None and hasattr(asp, "n"): return asp
#     gas = getattr(env, "get_action_space", None)
#     if callable(gas):
#         try:
#             out = gas()
#             if hasattr(out, "n"): return out
#             if isinstance(out, (list, tuple)): return SimpleNamespace(n=len(out))
#             if isinstance(out, int) and out > 0: return SimpleNamespace(n=out)
#         except Exception:
#             pass
#     if hasattr(env, "actions"):
#         try:
#             return SimpleNamespace(n=len(env.actions))
#         except Exception:
#             pass
#     return SimpleNamespace(n=5)

# def build_env_from_raw(raw_cfg: dict) -> G2RLEnv:
#     sig = inspect.signature(G2RLEnv.__init__)
#     allowed = {p.name for p in sig.parameters.values()
#                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
#     allowed.discard("self")
#     ctor_cfg = {k: v for k, v in raw_cfg.items() if k in allowed}
#     env = G2RLEnv(**ctor_cfg)
#     # 附带字段
#     try:
#         import numpy as _np
#         if "grid" in raw_cfg:
#             env.grid = (_np.array(raw_cfg["grid"]) > 0).astype(_np.uint8)
#     except Exception: pass
#     if "starts" in raw_cfg: env.starts = raw_cfg["starts"]
#     if "goals" in raw_cfg:  env.goals  = raw_cfg["goals"]
#     return env

# # ================== 复杂度合并 ==================
# def _merge_complexity_from_csv(base_map_settings: Dict[str, dict]) -> Dict[str, dict]:
#     if not os.path.exists(COMPLEXITY_CSV):
#         raise FileNotFoundError(f"未找到复杂度CSV：{COMPLEXITY_CSV}")

#     df = pd.read_csv(COMPLEXITY_CSV)

#     if "nn_complexity" not in df.columns:
#         if "nn_pred_success_rate" in df.columns:
#             df["nn_complexity"] = 1.0 - pd.to_numeric(df["nn_pred_success_rate"], errors="coerce")
#         else:
#             raise ValueError("复杂度CSV缺少 nn_complexity 且没有 nn_pred_success_rate 无法推导。")

#     df["nn_complexity"] = pd.to_numeric(df["nn_complexity"], errors="coerce")

#     csv_id_candidates  = [c for c in ["config_id", "grid_hash", "name", "map_id"] if c in df.columns]
#     yaml_id_candidates = set()
#     for spec in base_map_settings.values():
#         yaml_id_candidates.update(spec.keys())
#     yaml_id_candidates = [c for c in ["config_id", "grid_hash", "name", "map_id"] if c in yaml_id_candidates]
#     common_ids = [c for c in csv_id_candidates if c in yaml_id_candidates]
#     chosen_id = common_ids[0] if common_ids else None
#     use_composite = (chosen_id is None)

#     if not use_composite:
#         agg = df.groupby(chosen_id, dropna=False)["nn_complexity"].mean().reset_index().rename(
#             columns={"nn_complexity": "complexity"})
#         comp_map = dict(zip(agg[chosen_id].astype(str), agg["complexity"].astype(float)))
#     else:
#         needed = ["size", "num_agents", "density", "obs_radius", "max_episode_steps"]
#         missing = [c for c in needed if c not in df.columns]
#         if missing:
#             raise ValueError(f"无法用复合键匹配，CSV 缺少列：{missing}")
#         dfx = df.copy()
#         dfx["_size"] = pd.to_numeric(dfx["size"], errors="coerce").astype("Int64")
#         dfx["_nag"]  = pd.to_numeric(dfx["num_agents"], errors="coerce").astype("Int64")
#         dfx["_den"]  = pd.to_numeric(dfx["density"], errors="coerce").round(4)
#         dfx["_obs"]  = pd.to_numeric(dfx["obs_radius"], errors="coerce").astype("Int64")
#         dfx["_mep"]  = pd.to_numeric(dfx["max_episode_steps"], errors="coerce").astype("Int64")
#         comp = (
#             dfx.dropna(subset=["_size","_nag","_den","_obs","_mep"])
#                .groupby(["_size","_nag","_den","_obs","_mep"])["nn_complexity"]
#                .mean().reset_index().rename(columns={"nn_complexity":"complexity"})
#         )
#         comp_map = { (int(r["_size"]), int(r["_nag"]), float(r["_den"]), int(r["_obs"]), int(r["_mep"])): float(r["complexity"])
#                      for _, r in comp.iterrows() }

#     matched, unmatched = 0, 0
#     out: Dict[str, dict] = {}

#     for name, spec in base_map_settings.items():
#         new_spec = dict(spec)
#         cpx_val = np.nan

#         if not use_composite:
#             key = spec.get(chosen_id, None)
#             if key is None and chosen_id == "name":
#                 key = name
#             if key is not None:
#                 cpx_val = comp_map.get(str(key), np.nan)
#         else:
#             try:
#                 size  = int(spec.get("size"))
#                 nag   = int(spec.get("num_agents"))
#                 den   = float(spec.get("density"))
#                 obs   = int(spec.get("obs_radius"))
#                 mep   = int(spec.get("max_episode_steps"))
#                 tup   = (size, nag, round(den,4), obs, mep)
#                 cpx_val = comp_map.get(tup, np.nan)
#             except Exception:
#                 cpx_val = np.nan

#         if np.isfinite(cpx_val):
#             matched += 1
#             new_spec["complexity"] = float(cpx_val)
#         else:
#             unmatched += 1
#             new_spec["complexity"] = np.nan

#         out[name] = new_spec

#     print(f"🧩 复杂度匹配统计：matched={matched}, unmatched={unmatched}")
#     if unmatched > 0:
#         examples = [k for k,v in out.items() if not np.isfinite(v.get('complexity', np.nan))][:5]
#         print(f"⚠️ 有 {unmatched} 张地图未匹配到复杂度（未参与量化）。示例：{examples}")
#     return out

# # ================== 分位 + Stage2细化 ==================
# def _build_stages_refined(df: pd.DataFrame) -> List[dict]:
#     """
#     把复杂度按分位切成 5 档 ≈ 原 0..4，然后把原“Stage 2 桶”再一切为二 => 2a/2b
#     总阶段：0,1,2a,2b,3,4 => 共 6 阶
#     """
#     dfn = df.dropna(subset=["complexity"]).copy()
#     if len(dfn) == 0:
#         raise ValueError("没有可用地图用于 complexity 课程（complexity 全 NaN）。")
#     # 先切 5 桶
#     qs = np.linspace(0, 1, 6)  # 0..5 边界
#     edges = np.quantile(dfn["complexity"].values, qs)
#     buckets = []
#     for i in range(5):
#         lo, hi = float(edges[i]), float(edges[i+1]) + 1e-12
#         sub = dfn[(dfn["complexity"] >= lo) & (dfn["complexity"] < hi)].copy()
#         buckets.append((i, lo, hi, sub))

#     stages = []

#     # Stage 0
#     st0 = buckets[0]; stages.append({"stage":0, "cpx_min":st0[1], "cpx_max":st0[2], "items":st0[3].to_dict("records")})
#     # Stage 1
#     st1 = buckets[1]; stages.append({"stage":1, "cpx_min":st1[1], "cpx_max":st1[2], "items":st1[3].to_dict("records")})
#     # Stage 2 -> split into 2a/2b
#     st2 = buckets[2]
#     sub2 = st2[3].sort_values("complexity").reset_index(drop=True)
#     mid = max(1, len(sub2)//2)
#     sub2a = sub2.iloc[:mid]
#     sub2b = sub2.iloc[mid:]
#     stages.append({"stage":"2a", "cpx_min":float(sub2a["complexity"].min()), "cpx_max":float(sub2a["complexity"].max()), "items":sub2a.to_dict("records")})
#     stages.append({"stage":"2b", "cpx_min":float(sub2b["complexity"].min()), "cpx_max":float(sub2b["complexity"].max()), "items":sub2b.to_dict("records")})
#     # Stage 3
#     st3 = buckets[3]; stages.append({"stage":3, "cpx_min":st3[1], "cpx_max":st3[2], "items":st3[3].to_dict("records")})
#     # Stage 4
#     st4 = buckets[4]; stages.append({"stage":4, "cpx_min":st4[1], "cpx_max":st4[2], "items":st4[3].to_dict("records")})

#     return stages

# # ================== Scheduler ==================
# class ComplexityScheduler:
#     MAX_CONSEC_PER_MAP = 5
#     FAIL_STREAK_BAN    = 5
#     COOLDOWN_EPS       = 200

#     def __init__(self,
#                  base_map_settings: Dict[str, dict],
#                  min_per_stage: int = 8,
#                  min_episodes_per_stage: int = 300,
#                  threshold: float = 0.80,
#                  window_size: int = 120,
#                  shuffle_each_stage: bool = True,
#                  seed: int = 0):

#         self.min_episodes_per_stage = int(min_episodes_per_stage)
#         self.threshold = float(threshold)
#         self.window_size = int(window_size)

#         rows = [{"name": n, "complexity": s.get("complexity", np.nan), "spec": s}
#                 for n, s in base_map_settings.items()]
#         df = pd.DataFrame(rows)

#         stages = _build_stages_refined(df)

#         self._rng = random.Random(seed)
#         self._stage_items: List[List[dict]] = []
#         self._stage_edges: List[tuple] = []
#         self._stage_names: List[Union[int,str]] = []

#         for st in stages:
#             items = list(st["items"])
#             if shuffle_each_stage:
#                 self._rng.shuffle(items)
#             # 保底数量
#             if len(items) < min_per_stage:
#                 # 用全局最易地图补足
#                 all_sorted = df.dropna(subset=["complexity"]).sort_values("complexity").to_dict("records")
#                 need = min_per_stage - len(items)
#                 items += all_sorted[:need]
#             self._stage_items.append(items)
#             self._stage_edges.append((st["cpx_min"], st["cpx_max"]))
#             self._stage_names.append(st["stage"])

#         self.current_stage = 0
#         self.max_stage = len(self._stage_items) - 1

#         self._idx_in_stage = 0
#         self._win = deque(maxlen=self.window_size)
#         self._ep_in_stage = 0
#         self._succ_in_stage = 0
#         self._repeat_fail = 0
#         self._shrink_applied = set()

#         nS = self.max_stage + 1
#         self._pos = [0] * nS
#         self._sorted = [False] * nS
#         self._succ_streak = [0] * nS
#         self._fail_streak = [0] * nS

#         # 地图级统计
#         self._map_fail_streak = defaultdict(int)
#         self._map_fail_total  = defaultdict(int)
#         self._cooldown_until  = dict()   # ((stage_idx,name) -> until_ep)
#         self._current_item_name = None
#         self._map_state = defaultdict(lambda: {"consec": 0, "fail_streak": 0, "cooldown_until": -1})
#         self._global_ep = 0

#         self._ban_after = 18
#         self._banned = set()
#         self._ban_min_left = 3
#         self._ban_soft_stage_index = self._stage_names.index(1)  # stage<=1 不拉黑
#         self._long_cooldown = 300

#         # 细化策略：记录哪一层是 2a/2b
#         self._stage_is_2a = [ (self._stage_names[i] == "2a") for i in range(len(self._stage_names)) ]
#         self._stage_is_2b = [ (self._stage_names[i] == "2b") for i in range(len(self._stage_names)) ]

#         # 避坑带
#         self._cpx_trap_center = 0.102
#         self._cpx_trap_eps = 0.008

#         # 兜底轮换下标
#         self._easy_rotate = 0

#     # ---------- 过滤候选 ----------
#     def _eligible_pool(self, candidates, ep_idx):
#         pool = []
#         s_idx = self.current_stage
#         for r in candidates:
#             name = r["name"]
#             key = (s_idx, name)

#             cd_until = self._cooldown_until.get(key, -1)
#             if ep_idx < cd_until:  # 冷却中
#                 continue
#             if key in self._banned:
#                 continue

#             st = self._map_state[name]
#             if self._global_ep < st["cooldown_until"]:
#                 continue
#             if st["consec"] >= self.MAX_CONSEC_PER_MAP:
#                 st["cooldown_until"] = self._global_ep + self.COOLDOWN_EPS
#                 st["consec"] = 0
#                 continue

#             # Stage 2a/2b 的 agents 限制
#             spec = r.get("spec", {})
#             na = int(spec.get("num_agents", spec.get("agents", 99)))
#             if self._stage_is_2a[s_idx] and na > 4:
#                 continue
#             if self._stage_is_2b[s_idx] and na > 8:
#                 continue

#             # 避开 cpx 陷阱带
#             cpx = float(spec.get("complexity", 9e9)) if spec.get("complexity", None) is not None else 9e9
#             if abs(cpx - self._cpx_trap_center) <= self._cpx_trap_eps:
#                 # 只在池充足时过滤
#                 continue

#             pool.append(r)
#         return pool

#     # ---------- 兜底策略 ----------
#     def _fallback_pick_easy(self, items, pbar):
#         # 轮换最易前 K（自动跳过刚刚冷却/拉黑/agents超限）
#         s_idx = self.current_stage

#         def _ok(r):
#             name = r["name"]
#             key = (s_idx, name)
#             if key in self._banned: return False
#             cd_until = self._cooldown_until.get(key, -1)
#             if self._ep_in_stage < cd_until: return False
#             spec = r.get("spec", {})
#             na = int(spec.get("num_agents", spec.get("agents", 99)))
#             if self._stage_is_2a[s_idx] and na > 4: return False
#             if self._stage_is_2b[s_idx] and na > 8: return False
#             cpx = float(spec.get("complexity", 9e9)) if spec.get("complexity", None) is not None else 9e9
#             if abs(cpx - self._cpx_trap_center) <= self._cpx_trap_eps:
#                 return False
#             return True

#         easy_sorted = sorted(items, key=lambda r: r["spec"].get("complexity", 9e9))
#         K = min(8, max(3, len(easy_sorted)//3))
#         easy_sorted = [r for r in easy_sorted if _ok(r)]
#         if not easy_sorted:
#             # 实在没了，就允许略过陷阱带过滤
#             easy_sorted = sorted(items, key=lambda r: r["spec"].get("complexity", 9e9))[:K]

#         if not easy_sorted:
#             # 真没得选，解封最易
#             pick = min(items, key=lambda r: r["spec"].get("complexity", 9e9))
#             if pbar is not None:
#                 pbar.write(f"🛟 Stage {self._stage_names[s_idx]} 全部不可用，解封最易：{pick['name']}")
#             return pick

#         start = self._easy_rotate % len(easy_sorted)
#         pick = easy_sorted[start]
#         self._easy_rotate += 1
#         if pbar is not None:
#             pbar.write(f"🧯 Stage {self._stage_names[s_idx]} 兜底：候选均在冷却，轮换最易前{K} → {pick['name']}")
#         return pick

#     # ---------- 对外接口 ----------
#     def get_updated_map_settings(self, pbar=None) -> Dict[str, dict]:
#         if self.current_stage > self.max_stage:
#             return {}

#         s_idx = self.current_stage
#         items = self._stage_items[s_idx]
#         if not items:
#             raise RuntimeError(f"Stage {self._stage_names[s_idx]} 没有地图。")

#         if not self._sorted[s_idx]:
#             def _key(r):
#                 c = r["spec"].get("complexity", None)
#                 return (1, float("inf")) if (c is None or not np.isfinite(c)) else (0, float(c))
#             items = sorted(items, key=_key)
#             self._stage_items[s_idx] = items
#             self._pos[s_idx] = 0
#             self._sorted[s_idx] = True

#         ep_idx = self._ep_in_stage
#         usable = self._eligible_pool(items, ep_idx)

#         if not usable:
#             pick = self._fallback_pick_easy(items, pbar)
#         else:
#             idx = max(0, min(self._pos[s_idx], len(usable) - 1))
#             pick = usable[idx]

#         self._current_item_name = pick["name"]
#         return {pick["name"]: pick["spec"]}

#     def add_episode_result(self, success: int, pbar=None):
#         s_val = 1 if success else 0
#         self._win.append(s_val)
#         self._ep_in_stage += 1
#         self._succ_in_stage += s_val

#         s_idx = self.current_stage
#         L = len(self._stage_items[s_idx])
#         self._pos[s_idx] = max(0, min(self._pos[s_idx], max(L - 1, 0)))

#         if success:
#             self._succ_streak[s_idx] += 1
#             self._fail_streak[s_idx] = 0
#             if self._succ_streak[s_idx] >= 1:
#                 self._pos[s_idx] = min(self._pos[s_idx] + 1, L - 1)
#                 self._succ_streak[s_idx] = 0
#         else:
#             self._fail_streak[s_idx] += 1
#             self._succ_streak[s_idx] = 0
#             if self._fail_streak[s_idx] >= 1:
#                 self._pos[s_idx] = max(self._pos[s_idx] - 1, 0)
#                 self._fail_streak[s_idx] = 0

#         # 同图连败冷却/拉黑
#         name = self._current_item_name
#         if name is None: return
#         key = (s_idx, name)

#         st = self._map_state[name]
#         st["consec"] += 1
#         if success:
#             st["fail_streak"] = 0
#         else:
#             st["fail_streak"] += 1
#             if st["fail_streak"] >= self.FAIL_STREAK_BAN:
#                 st["cooldown_until"] = self._global_ep + self.COOLDOWN_EPS
#                 st["consec"] = 0

#         cpx = None
#         try:
#             cur = next((r for r in self._stage_items[s_idx] if r["name"] == name), None)
#             cpx = cur["spec"].get("complexity", None) if cur else None
#         except Exception:
#             pass

#         # 针对陷阱带 cpx≈0.102
#         if (not success) and (cpx is not None) and np.isfinite(cpx) and abs(float(cpx)-0.102) <= 0.008:
#             self._pos[s_idx] = max(self._pos[s_idx] - 2, 0)
#             self._cooldown_until[key] = self._ep_in_stage + 180  # 更长冷却
#             if pbar is not None:
#                 pbar.write(f"⛔ cpx≈0.102 快速退避：Stage {self._stage_names[s_idx]} 回退2格并冷却 {name}")
#             self._global_ep += 1
#             return

#         self._map_fail_streak[key] += (0 if success else 1)
#         if not success:
#             self._map_fail_total[key] += 1

#         if (not success) and self._map_fail_streak[key] >= 3:
#             self._pos[s_idx] = max(self._pos[s_idx] - 1, 0)
#             self._cooldown_until[key] = self._ep_in_stage + 120  # 冷却更久
#             self._map_fail_streak[key] = 0
#             if pbar is not None:
#                 pbar.write(f"📉 连败触发：Stage {self._stage_names[s_idx]} 回退一格并冷却 {name}")

#         if self._map_fail_total[key] >= self._ban_after:
#             if s_idx <= self._ban_soft_stage_index:
#                 self._cooldown_until[key] = self._ep_in_stage + 300
#                 self._map_fail_streak[key] = 0
#                 self._map_fail_total[key]  = 0
#                 if pbar is not None:
#                     pbar.write(f"🧊 Stage {self._stage_names[s_idx]} 保护：低阶段不拉黑，冷却 {name}（300 ep）")
#             else:
#                 remain = sum((s_idx, r["name"]) not in self._banned for r in self._stage_items[s_idx])
#                 if remain <= self._ban_min_left:
#                     self._cooldown_until[key] = self._ep_in_stage + 300
#                     self._map_fail_streak[key] = 0
#                     self._map_fail_total[key]  = 0
#                     if self._pos[s_idx] > 0:
#                         self._pos[s_idx] -= 1
#                     if pbar is not None:
#                         pbar.write(f"🛡️ 保护：Stage {self._stage_names[s_idx]} 可用≤{self._ban_min_left}，不拉黑 {name}，改冷却 300 ep")
#                 else:
#                     self._banned.add(key)
#                     if self._pos[s_idx] > 0:
#                         self._pos[s_idx] -= 1
#                     self._map_fail_streak[key] = 0
#                     self._map_fail_total[key]  = 0
#                     msg = f"🚫 Stage {self._stage_names[s_idx]} 拉黑地图：{name}（累计失败≥{self._ban_after}）"
#                     if pbar is not None: pbar.write(msg)
#                     else: print(msg)

#         self._global_ep += 1

#     def window_sr(self) -> float:
#         return float(sum(self._win) / len(self._win)) if len(self._win) else 0.0

#     def stage_sr(self) -> float:
#         return float(self._succ_in_stage / max(1, self._ep_in_stage))

#     def should_advance(self) -> bool:
#         if self._ep_in_stage < self.min_episodes_per_stage:
#             return False
#         return self.stage_sr() >= self.threshold

#     def advance(self, pbar=None):
#         lo, hi = self._stage_edges[self.current_stage]
#         name = self._stage_names[self.current_stage]
#         if pbar is not None:
#             pbar.write(f"✅ 通过 Stage {name} | SR(win)={self.window_sr():.2f} / SR(stage)={self.stage_sr():.2f} | 区间[{lo:.4f}, {hi:.4f}] → 下一阶段")
#         self.current_stage += 1
#         self._reset_stage_stats()
#         self._repeat_fail = 0

#     def repeat_stage(self, pbar=None):
#         self._repeat_fail += 1
#         if pbar is not None:
#             pbar.write(
#                 f"🔁 未达标，重复 Stage {self._stage_names[self.current_stage]}（已训练 {self._ep_in_stage} ep，SR={self.stage_sr():.2f}，连续失败={self._repeat_fail}）"
#             )
#         # 多次失败—缩池（保留更易的前 30%）
#         if (self._repeat_fail >= 2) and (self.current_stage not in self._shrink_applied):
#             items = self._stage_items[self.current_stage]
#             if items:
#                 def _cmp_key(r):
#                     c = r['spec'].get('complexity', None)
#                     return (1, 1e9) if (c is None or not np.isfinite(c)) else (0, float(c))
#                 items_sorted = sorted(items, key=_cmp_key)
#                 keep = max(4, len(items_sorted) // 3)   # 只留最易前 1/3
#                 new_pool = items_sorted[:keep]
#                 self._stage_items[self.current_stage] = new_pool
#                 self._shrink_applied.add(self.current_stage)
#                 if pbar is not None:
#                     pbar.write(f"📉 降难：Stage {self._stage_names[self.current_stage]} 地图池 {len(items)} → {len(new_pool)}（保留最易前 30%）")
#                 self._idx_in_stage = 0
#         self._reset_stage_stats()

#     def _reset_stage_stats(self):
#         self._idx_in_stage = 0
#         self._win.clear()
#         self._ep_in_stage = 0
#         self._succ_in_stage = 0
#         if 0 <= self.current_stage <= self.max_stage:
#             s = self.current_stage
#             self._pos[s] = 0
#             self._succ_streak[s] = 0
#             self._fail_streak[s] = 0
#             self._sorted[s] = False

#     def is_done(self) -> bool:
#         return self.current_stage > self.max_stage

#     def current_stage_name(self):
#         if 0 <= self.current_stage < len(self._stage_names):
#             return self._stage_names[self.current_stage]
#         return "done"

# # ================== 训练 ==================
# def get_timestamp() -> str:
#     return datetime.now().strftime('%H-%M-%d-%m-%Y')

# def _pogema_num_agents_from_obs(obs) -> int:
#     try:
#         return int(len(obs))
#     except Exception:
#         return 1

# def _idle_action_index(env, default_idx=0) -> int:
#     if hasattr(env, "actions") and isinstance(env.actions, (list, tuple)):
#         for i, name in enumerate(env.actions):
#             if isinstance(name, str) and name.lower() in ("idle", "stay", "noop", "wait"):
#                 return i
#     return default_idx

# def opposite_of(prev_action: int, idx2move: dict, idle_idx: int) -> set:
#     v = idx2move.get(prev_action, None)
#     if v is None or v == (0, 0):
#         return set()
#     inv = (-v[0], -v[1])
#     return {a for a, mv in idx2move.items() if mv == inv}

# def will_conflict_next_step(obs_prev, actions, target_idx: int, idx2move: dict) -> bool:
#     try:
#         n = len(obs_prev)
#         cur = [tuple(obs_prev[i]['global_xy']) for i in range(n)]
#     except Exception:
#         return False

#     nxt = []
#     for i in range(len(actions)):
#         a = actions[i]
#         dx, dy = idx2move.get(a, (0, 0))
#         x, y = cur[i]
#         nxt.append((x + dx, y + dy))

#     tgt_next = nxt[target_idx]
#     if any((i != target_idx and nxt[i] == tgt_next) for i in range(len(nxt))):
#         return True

#     tgt_cur = cur[target_idx]
#     for i in range(len(nxt)):
#         if i == target_idx: continue
#         if nxt[i] == tgt_cur and nxt[target_idx] == cur[i]:
#             return True
#     return False

# def local_congestion(obs, idx2move, radius=1) -> int:
#     """
#     粗略估计局部拥堵：统计半径 r 内的其他智能体数（用于塑形）
#     """
#     try:
#         centers = [tuple(o['global_xy']) for o in obs]
#     except Exception:
#         return 0
#     cx, cy = centers[0] if len(centers) else (0,0)
#     cnt = 0
#     for i,(x,y) in enumerate(centers[1:], start=1):
#         if abs(x-cx) + abs(y-cy) <= radius:
#             cnt += 1
#     return cnt

# def _invalid_actions_for(env, obs_i, idle_idx=0):
#     invalid = []
#     try:
#         x, y = obs_i['global_xy']
#     except Exception:
#         return invalid, {idle_idx:(0,0)}

#     moves = {
#         "up":    (-1, 0),
#         "right": (0, 1),
#         "down":  (1, 0),
#         "left":  (0, -1),
#     }
#     mapping = {}
#     if hasattr(env, "actions") and isinstance(env.actions, (list, tuple)):
#         for idx, name in enumerate(env.actions):
#             if isinstance(name, str):
#                 low = name.lower()
#                 if low in ("idle", "stay", "noop", "wait"):
#                     mapping[idx] = (0, 0)
#                 elif "up" in low:
#                     mapping[idx] = moves["up"]
#                 elif "right" in low:
#                     mapping[idx] = moves["right"]
#                 elif "down" in low:
#                     mapping[idx] = moves["down"]
#                 elif "left" in low:
#                     mapping[idx] = moves["left"]
#     if not mapping:
#         mapping = {idle_idx:(0,0), 1:(-1,0), 2:(0,1), 3:(1,0), 4:(0,-1)}

#     grid = getattr(env, "grid", None)
#     H, W = (grid.shape[:2] if grid is not None else (None, None))
#     for a, (dx, dy) in mapping.items():
#         nx, ny = x + dx, y + dy
#         if H is not None and (nx < 0 or ny < 0 or nx >= H or ny >= W):
#             invalid.append(a); continue
#         if grid is not None:
#             try:
#                 if grid[nx, ny] != 0:
#                     invalid.append(a); continue
#             except Exception:
#                 pass
#     return list(set(invalid)), mapping

# def _rb_is_iterable(rb) -> bool:
#     try:
#         iter(rb)
#         return True
#     except TypeError:
#         return False
#     except Exception:
#         return False

# def train(
#     model: torch.nn.Module,
#     map_settings: Dict[str, dict],
#     map_probs: Union[List[float], None],
#     num_episodes: int = 300,
#     batch_size: int = 128,
#     decay_range: int = 1000,
#     log_dir='logs',
#     lr: float = 0.001,
#     replay_buffer_size: int = 1000,
#     device: str = 'cuda',
#     scheduler: Optional[ComplexityScheduler] = None,
#     max_episode_seconds: int = 60,
#     run_dir: Optional[str] = None
# ) -> DDQNAgent:

#     timestamp = get_timestamp()
#     run_dir = Path(log_dir) / timestamp if run_dir is None else Path(run_dir)
#     run_dir.mkdir(parents=True, exist_ok=True)
#     writer = SummaryWriter(log_dir=str(run_dir))

#     # —— 阶段超参覆盖（细化 2a/2b）
#     STAGE_OVERRIDES = {
#         0:  {"max_episode_seconds": 45, "batch_size": 48,  "replay_buffer_size":  8000, "target_tau": 0.02, "eps_min": 0.10},
#         1:  {"max_episode_seconds": 60, "batch_size": 64,  "replay_buffer_size": 16000, "target_tau": 0.02, "eps_min": 0.15},
#         # 注意：索引与 scheduler.current_stage 对齐，名字是 2a/2b 但这里按索引填
#         2:  {"max_episode_seconds": 95, "batch_size": 96,  "replay_buffer_size": 32000, "target_tau": 0.04, "eps_min": 0.30},  # 2a
#         3:  {"max_episode_seconds": 90, "batch_size": 96,  "replay_buffer_size": 32000, "target_tau": 0.035,"eps_min": 0.25},  # 2b
#         4:  {"max_episode_seconds": 90, "batch_size": 128, "replay_buffer_size": 40000, "target_tau": 0.03, "eps_min": 0.18},
#         5:  {"max_episode_seconds": 95, "batch_size": 128, "replay_buffer_size": 50000, "target_tau": 0.03, "eps_min": 0.12},
#     }

#     # —— 初始化第一个 env & 动作空间
#     first_name = next(iter(map_settings))
#     first_env = build_env_from_raw(map_settings[first_name])
#     try:
#         first_env.reset()
#     except Exception:
#         pass

#     try:
#         asp = _safe_get_action_space(first_env)
#         n_actions = int(getattr(asp, "n", 5))
#     except Exception:
#         n_actions = 5
#     action_space_list = list(range(n_actions))
#     print(f"✅ n_actions = {n_actions}")

#     # 预热模型
#     try:
#         obs0, _ = first_env.reset()
#     except Exception:
#         obs0 = first_env.reset()
#     state0 = obs0[0] if isinstance(obs0, (list, tuple)) else obs0
#     in_channels = 11
#     with torch.no_grad():
#         _ = model(_obs_to_tensor_CDHW(state0, device, expected_c=in_channels))
#     print(f"🔥 warmed model with in_channels={in_channels}")

#     def preprocess_single(s):
#         return _obs_to_tensor_CDHW(s, device, expected_c=in_channels).squeeze(0)

#     # 构造 Agent
#     agent = DDQNAgent(
#         model, model, action_space_list,
#         lr=lr, decay_range=decay_range, device=device,
#         replay_buffer_size=replay_buffer_size,
#         obs_preprocessor=preprocess_single,
#     )
#     if not hasattr(agent, "success_buffer"):
#         agent.success_buffer = deque(maxlen=8000)  # 放大成功缓存

#     training_logs = []
#     pbar = tqdm(desc='Episodes', unit='ep', dynamic_ncols=True)

#     episode = 0
#     success_count_total = 0

#     while True:
#         if scheduler is None:
#             raise ValueError("需要 scheduler 才能按阶段课程训练直到全部完成。")
#         if scheduler.is_done():
#             break

#         # 1) 当前阶段地图 & 构建 env
#         cur_map_cfg = scheduler.get_updated_map_settings(pbar=pbar)
#         map_name, cfg = next(iter(cur_map_cfg.items()))
#         env = build_env_from_raw(cfg)

#         stage_idx = scheduler.current_stage
#         stage_name = scheduler.current_stage_name()
#         cpx_val = cfg.get("complexity", None)
#         if cpx_val is not None and np.isfinite(cpx_val):
#             _pwrite(pbar, f"🟢 使用地图：{map_name} | Stage {stage_name} | Agents={env.num_agents} | Complexity={float(cpx_val):.3f}")
#         else:
#             _pwrite(pbar, f"🟢 使用地图：{map_name} | Stage {stage_name} | Agents={env.num_agents}")

#         # 2) reset
#         try:
#             obs, info = env.reset()
#         except Exception:
#             obs = env.reset(); info = {}

#         # 阶段覆盖参数
#         ovr = STAGE_OVERRIDES.get(stage_idx, {})
#         used_batch_size   = int(ovr.get("batch_size", batch_size))
#         used_max_seconds  = int(ovr.get("max_episode_seconds", max_episode_seconds))
#         used_eps_min      = float(ovr.get("eps_min", 0.1))
#         target_tau        = float(ovr.get("target_tau", 0.02))

#         # 动态LR（Stage>=3 ×1.2；Stage 2a/2b ×1.1）
#         if hasattr(agent, "optimizer"):
#             if not hasattr(agent, "_base_lrs"):
#                 agent._base_lrs = [pg["lr"] for pg in agent.optimizer.param_groups]
#             for g, lr0 in zip(agent.optimizer.param_groups, agent._base_lrs):
#                 mul = 1.2 if stage_idx >= 4 else (1.1 if stage_idx in (2,3) else 1.0)
#                 g["lr"] = lr0 * mul

#         # tau
#         for attr in ["tau","target_tau","soft_update_tau","target_update_tau"]:
#             if hasattr(agent, attr):
#                 setattr(agent, attr, float(target_tau))
#                 break

#         # 扩容经验池
#         if hasattr(agent, "resize_buffer") and callable(agent.resize_buffer):
#             agent.resize_buffer(int(ovr.get("replay_buffer_size", replay_buffer_size)))
#         elif hasattr(agent, "replay_buffer") and hasattr(agent.replay_buffer, "maxlen"):
#             tmp = list(agent.replay_buffer)
#             agent.replay_buffer = deque(tmp, maxlen=int(ovr.get("replay_buffer_size", replay_buffer_size)))

#         # Epsilon 下限
#         if hasattr(agent, "epsilon"):
#             agent.epsilon = max(used_eps_min, float(getattr(agent, "epsilon", used_eps_min)))

#         if stage_idx in (2,3):
#             _pwrite(pbar, f"🛠️ Stage {stage_name} overrides: batch_size={used_batch_size}, max_seconds={used_max_seconds}, tau={target_tau}, eps_min={used_eps_min}")
#             _pwrite(pbar, f"🗃️ Stage {stage_name}: replay buffer → {ovr.get('replay_buffer_size', replay_buffer_size)}")

#         # 观测长度作为代理数
#         n_agents_pg = _pogema_num_agents_from_obs(obs)
#         idle_idx = _idle_action_index(env, 0)
#         invalid_0, idx2move = _invalid_actions_for(env, obs[0], idle_idx)

#         # 目标 agent：Stage<=1 取最短距目标；2a/2b 仍取最短（更友好）；后续可换中位
#         try:
#             dists = [(abs(obs[i]['global_xy'][0] - env.goals[i][0]) +
#                       abs(obs[i]['global_xy'][1] - env.goals[i][1]), i)
#                      for i in range(n_agents_pg)]
#             target_idx = min(dists)[1]
#         except Exception:
#             target_idx = int(np.random.randint(n_agents_pg))

#         # 队友：pogema A* or 内置 Greedy
#         teammates = [None] * n_agents_pg
#         for i in range(n_agents_pg):
#             if i == target_idx: continue
#             if HAVE_POGEMA: teammates[i] = _PogemaAStarAgent()
#             else:           teammates[i] = GreedyAgent(env, idle_idx, idx2move)

#         # 目标 & 初始状态
#         try:
#             goal = tuple(env.goals[target_idx])
#         except Exception:
#             try:    goal = tuple(obs[target_idx]["global_goal"])
#             except: goal = tuple(obs[target_idx]["global_xy"])
#         state = obs[target_idx]

#         # 动态步数上限：最短路估计 + 阶段系数
#         try:
#             opt_len = max(1, len(env.global_guidance[target_idx]) + 1)
#         except Exception:
#             try:
#                 sx, sy = state['global_xy']; gx, gy = goal
#                 opt_len = max(1, abs(sx - gx) + abs(sy - gy) + 1)
#             except Exception:
#                 opt_len = 60

#         cfg_max = int(cfg.get("max_episode_steps", 10**9))
#         env_max = int(getattr(env, "max_episode_steps", cfg_max))
#         mult = 9 if stage_idx <= 1 else (12 if stage_idx in (2,3) else 12)
#         timesteps_per_episode = min(env_max, max(100, int(opt_len * mult)))

#         # 陷阱带放宽时间
#         if cpx_val is not None and np.isfinite(cpx_val) and abs(float(cpx_val) - 0.102) <= 0.008:
#             timesteps_per_episode = int(timesteps_per_episode * 1.3)
#             used_max_seconds = max(used_max_seconds, 95)

#         # —— 开始一集
#         model.eval()
#         episode_traj = []
#         episode_start_time = time.time()
#         success_flag = False
#         no_move_steps = 0
#         last_pos_for_stuck = tuple(state.get('global_xy', (None, None))) if isinstance(state, dict) else None
#         prev_action = None

#         for t in range(timesteps_per_episode):
#             if time.time() - episode_start_time > used_max_seconds:
#                 _pwrite(pbar, f"⏰ Episode {episode} 超时（>{used_max_seconds}s），终止本集")
#                 break

#             prev_pos = tuple(state.get('global_xy', (0, 0))) if isinstance(state, dict) else None

#             actions = [idle_idx] * n_agents_pg
#             for i in range(n_agents_pg):
#                 if i == target_idx:
#                     # —— 风险感知 Q 选择
#                     x = _obs_to_tensor_CDHW(obs[i], device, expected_c=in_channels)  # [1,C,D,H,W]
#                     eps = float(getattr(agent, "epsilon", 0.1))
#                     if random.random() < eps:
#                         a = random.randrange(n_actions)
#                     else:
#                         with torch.no_grad():
#                             q = agent.q_network(x)  # [1, A]
#                             invalid, idx2move = _invalid_actions_for(env, obs[i], idle_idx)
#                             if invalid:
#                                 q[0, invalid] = -1e9

#                             # 预判冲突/边交换：风险动作打折
#                             risk_mask = []
#                             for a_idx in range(n_actions):
#                                 am = actions[:]  # 拷贝
#                                 am[i] = a_idx
#                                 if will_conflict_next_step(obs, am, target_idx, idx2move):
#                                     risk_mask.append(a_idx)
#                             if risk_mask:
#                                 # 打 0.7 折（也可以直接屏蔽成 -1e9）
#                                 q[0, risk_mask] *= 0.7

#                             a = int(torch.argmax(q, dim=1))
#                     actions[i] = a
#                 else:
#                     try:
#                         if HAVE_POGEMA: a_tm = int(teammates[i].act(obs[i]))
#                         else:           a_tm = int(teammates[i].act(obs[i], env.goals[i]))
#                     except Exception:
#                         a_tm = idle_idx
#                     if random.random() < 0.05:  # 小扰动
#                         a_tm = idle_idx
#                     actions[i] = a_tm

#             # —— 奖励塑形（执行前的预测项）
#             osc_pen = 0.0
#             pre_collision_pen = 0.0
#             try:
#                 if prev_action is not None and actions[target_idx] in opposite_of(prev_action, idx2move, idle_idx):
#                     osc_pen = -0.05 if stage_idx in (2,3) else -0.03
#             except Exception: pass

#             try:
#                 if will_conflict_next_step(obs, actions, target_idx, idx2move):
#                     pre_collision_pen = -0.20 if stage_idx in (2,3) else -0.12
#             except Exception: pass

#             prev_action = actions[target_idx]  # 保存动作

#             # —— 真正执行一步
#             obs, reward, terminated, truncated, info = env.step(tuple(int(a) for a in actions))

#             # —— 拥堵/等待塑形（执行后的加成）
#             try:
#                 # 稍轻时间罚
#                 r_time = -0.0005 if stage_idx in (2,3) else -0.001

#                 # 稠密引导
#                 ax, ay = obs[target_idx]['global_xy']; gx, gy = goal
#                 if prev_pos is not None:
#                     px, py = prev_pos
#                     d_prev = abs(px - gx) + abs(py - gy)
#                     d_now  = abs(ax - gx) + abs(ay - gy)
#                     r_dense = 0.05 * (d_prev - d_now)
#                 else:
#                     r_dense = 0.0

#                 # 碰撞罚（若有）
#                 collided = False
#                 try:
#                     if isinstance(info, dict):
#                         col = info.get("collision", None)
#                         if isinstance(col, (list, tuple)):
#                             collided = bool(col[target_idx])
#                         elif isinstance(col, dict):
#                             collided = bool(col.get(target_idx, False))
#                 except Exception:
#                     pass
#                 r_coll = -0.2 if collided else 0.0

#                 # 拥堵罚（曼哈顿半径1内人多则小罚），等待奖（选择 idle 奖励）
#                 cong = local_congestion(obs, idx2move, radius=1)
#                 r_cong = -0.02 * max(0, cong - 1)     # 多于1人开始罚
#                 chose_idle = (actions[target_idx] == idle_idx)
#                 wait_bonus = (0.06 if stage_idx in (2,3) else 0.03) if chose_idle else 0.0

#                 reward[target_idx] += (osc_pen + pre_collision_pen + r_dense + r_coll + r_time + r_cong + wait_bonus)
#             except Exception:
#                 pass

#             # 到达判定
#             try:
#                 agent_pos = tuple(obs[target_idx]['global_xy'])
#             except Exception:
#                 agent_pos = tuple(state.get('global_xy', (0, 0)))
#             done = (agent_pos == goal)
#             terminated[target_idx] = done

#             # STUCK 终止
#             if STUCK_PATIENCE > 0:
#                 if last_pos_for_stuck == agent_pos:
#                     no_move_steps += 1
#                     if no_move_steps >= STUCK_PATIENCE: break
#                 else:
#                     no_move_steps = 0
#                     last_pos_for_stuck = agent_pos

#             trans = (state, actions[target_idx], reward[target_idx], obs[target_idx], terminated[target_idx])
#             agent.store(*trans)
#             state = obs[target_idx]
#             episode_traj.append(trans)

#             if done:
#                 success_flag = True
#                 break

#             # —— 经验回放训练（支持 PER/非PER）
#             try:
#                 rb = getattr(agent, "replay_buffer", None)
#                 if rb is None:
#                     pass
#                 elif _rb_is_iterable(rb):
#                     rb_list = list(rb)
#                     n = len(rb_list)
#                     if n >= used_batch_size:
#                         # Stage2 系列混采比：成功60 / 近期35 / 困难5
#                         if stage_idx in (2,3):
#                             n_succ, n_recent = int(used_batch_size*0.60), int(used_batch_size*0.35)
#                             n_hard = used_batch_size - n_succ - n_recent
#                             recent_cap = 12000
#                         else:
#                             n_succ, n_recent = int(used_batch_size*0.30), used_batch_size - int(used_batch_size*0.30)
#                             n_hard = 0
#                             recent_cap = 8000

#                         split = max(0, n - recent_cap)
#                         older, recent = rb_list[:split], rb_list[split:]
#                         batch = []

#                         if recent:
#                             batch += random.sample(recent, min(len(recent), n_recent))
#                         if len(batch) < n_recent and older:
#                             need = n_recent - len(batch)
#                             batch += random.sample(older, min(len(older), need))

#                         if hasattr(agent, "success_buffer") and agent.success_buffer and n_succ > 0:
#                             succ_pool = list(agent.success_buffer)
#                             if succ_pool:
#                                 batch += random.sample(succ_pool, min(len(succ_pool), n_succ))

#                         def _dist_to_goal_state(st):
#                             try:
#                                 ax, ay = st['global_xy']; gx, gy = goal
#                                 return abs(ax-gx)+abs(ay-gy)
#                             except Exception:
#                                 return 1e9

#                         if n_hard > 0:
#                             pool_all = (recent or []) + (older or [])
#                             pool_all.sort(key=lambda tr: _dist_to_goal_state(tr[0]))
#                             hard = []
#                             for tr in pool_all:
#                                 done_flag = bool(tr[4])
#                                 if not done_flag:
#                                     hard.append(tr)
#                                 if len(hard) >= n_hard: break
#                             batch += hard

#                         random.shuffle(batch)
#                         batch = batch[:used_batch_size]
#                         if len(batch) >= used_batch_size:
#                             if hasattr(agent, "retrain_from_transitions") and callable(agent.retrain_from_transitions):
#                                 _ = agent.retrain_from_transitions(batch)
#                             else:
#                                 _ = agent.retrain(used_batch_size)
#                 else:
#                     if hasattr(rb, "__len__") and len(rb) >= used_batch_size:
#                         _ = agent.retrain(used_batch_size)
#             except Exception:
#                 try:
#                     if hasattr(agent, "replay_buffer") and len(agent.replay_buffer) >= used_batch_size:
#                         _ = agent.retrain(used_batch_size)
#                 except Exception:
#                     pass

#         # 成功缓存
#         if success_flag and hasattr(agent, "success_buffer"):
#             for t_ in episode_traj:
#                 agent.success_buffer.append(t_)

#         # 统计 & 日志
#         if success_flag:
#             success_count_total += 1
#         episode += 1

#         scheduler.add_episode_result(int(success_flag), pbar)
#         sr_global = success_count_total / max(1, episode)
#         sr_stage_now = scheduler.stage_sr()

#         writer.add_scalar('SuccessRate/global', sr_global, episode)
#         writer.add_scalar('success', 1 if success_flag else 0, episode)
#         writer.add_scalar('SuccessRate/stage', sr_stage_now, episode)

#         if pbar is not None:
#             pbar.set_postfix(
#                 Stage=str(stage_name),
#                 success=int(success_flag),
#                 sr_global=f"{sr_global:.3f}",
#                 sr_stage=f"{sr_stage_now:.3f}"
#             )
#             pbar.update(1)

#         print(f"[Episode {episode}] Stage {stage_name} | Success={'✅' if success_flag else '❌'} | "
#               f"SR_global={sr_global:.3f}, SR_stage={sr_stage_now:.6f}")

#         training_logs.append({
#             "episode": episode,
#             "stage": str(stage_name),
#             "map": map_name,
#             "agents": _pogema_num_agents_from_obs(obs),
#             "complexity": (float(cpx_val) if cpx_val is not None else np.nan),
#             "success": int(success_flag),
#             "success_rate_global": float(sr_global),
#             "success_rate_stage": float(sr_stage_now),
#         })

#         # 晋级/重复
#         if scheduler.should_advance():
#             status_line = f"✅ 通过 Stage {stage_name} | SR(win)={scheduler.window_sr():.2f} / SR(stage)={scheduler.stage_sr():.2f}"
#             _pwrite(pbar, status_line)
#             try:
#                 ckpt_path = Path("models") / f"stage_{stage_name}_model.pt"
#                 ckpt_path.parent.mkdir(parents=True, exist_ok=True)
#                 torch.save(model.state_dict(), ckpt_path.as_posix())
#                 _pwrite(pbar, f"💾 阶段权重已保存：{ckpt_path}")
#             except Exception as e:
#                 _pwrite(pbar, f"⚠️ 保存阶段权重失败：{e}")
#             try:
#                 log_file = Path(run_dir) / "stage_transitions.log"
#                 with open(log_file, "a", encoding="utf-8") as f:
#                     f.write(status_line + "\n")
#             except Exception: pass
#             scheduler.advance(pbar)
#             # 进入新阶段：epsilon 回弹到该阶段下限
#             if hasattr(agent, "epsilon"):
#                 agent.epsilon = STAGE_OVERRIDES.get(scheduler.current_stage, {}).get("eps_min", agent.epsilon)
#             continue
#         else:
#             if scheduler._ep_in_stage >= scheduler.min_episodes_per_stage:
#                 scheduler.repeat_stage(pbar)

#     # —— 收尾
#     df = pd.DataFrame(training_logs)
#     out_dir = Path(run_dir) / "episodes"
#     out_dir.mkdir(parents=True, exist_ok=True)
#     csv_path = out_dir / "episodes.csv"
#     df.to_csv(csv_path, index=False, encoding="utf-8-sig")
#     print(f"📝 每集日志已保存：{csv_path}")
#     _pwrite(pbar, "🎉 所有阶段训练完成！")
#     writer.close()
#     return agent

# # ================== 主程序 ==================
# if __name__ == "__main__":
#     os.makedirs('logs', exist_ok=True)
#     os.makedirs('models', exist_ok=True)

#     # 1) 读 YAML
#     with open(MAP_SETTINGS_PATH, "r", encoding="utf-8") as f:
#         base_map_settings = yaml.safe_load(f)
#     if isinstance(base_map_settings, list):
#         base_map_settings = {(m.get("name") or f"map_{i}"): m for i, m in enumerate(base_map_settings)}

#     # 2) 合并复杂度
#     base_map_settings = _merge_complexity_from_csv(base_map_settings)

#     # 3) 构建 Scheduler（细化后共 6 阶）
#     scheduler = ComplexityScheduler(
#         base_map_settings=base_map_settings,
#         min_per_stage=MIN_PER_STAGE,
#         min_episodes_per_stage=MIN_EPISODES_PER_STAGE,
#         threshold=THRESHOLD,
#         window_size=WINDOW_SIZE,
#         shuffle_each_stage=True,
#         seed=0,
#     )

#     # 4) 设备 & 模型
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = CRNNModel().to(device)

#     # 5) 训练（注意：map_settings 只给一个占位，后续每集 scheduler 会切换）
#     init_map = scheduler.get_updated_map_settings()
#     agent = train(
#         model=model,
#         scheduler=scheduler,
#         map_settings=init_map,
#         map_probs=None,
#         batch_size=BATCH_SIZE,
#         replay_buffer_size=REPLAY_BUFFER_SIZE,
#         decay_range=DECAY_RANGE,
#         log_dir=LOG_DIR,
#         device=device,
#         max_episode_seconds=MAX_EPISODE_SECONDS,
#         run_dir=RUN_DIR,
#     )

#     # 6) 保存最终模型
#     out_path = Path(MODEL_OUT)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     torch.save(model.state_dict(), out_path.as_posix())
#     print(f"✅ 模型已保存到 {out_path}")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train0829.py — 分层课程(细化) + 风险感知 + 拥堵/等待塑形 + 冷却/拉黑 + 兜底避坑
兼容用户工程：G2RLEnv / DDQNAgent / CRNNModel

关键特性：
1) 课程细化：原 5 桶（0..4），对 2/3/4 各自再切半 => 0,1,2a,2b,3a,3b,4a,4b 共 8 阶。
2) 子阶段 Agents 上限：2a≤4，2b/3a≤8，3b/4a≤12，4b≤16；训练更稳。
3) 风险感知动作：预判顶点/边冲突，给风险动作打折；拥堵/等待塑形（主动让路奖、拥堵罚、时间罚）。
4) 经验回放混采：低→高阶段逐步加大“成功/近期/困难”的配比，Stage 越高“困难”比例越高。
5) 地图调度：同图冷却、连败回退、黑名单（带保底）、避开 cpx≈0.102 坑带；全冷却时兜底轮换“最易前K”。
6) tqdm 安全打印：严禁对 tqdm 对象做 bool 判断；统一 _pwrite；兜底挑图防止 IndexError。
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union, List, Tuple
from types import SimpleNamespace
import os, sys, time, inspect, random
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm
from collections import deque, defaultdict

# ===================== 配置 =====================
# 1) YAML 地图配置
MAP_SETTINGS_PATH = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main - train/g2rl/map_settings_generated_new.yaml"

# 2) 复杂度 CSV（由 infer_complexity.py 生成）
COMPLEXITY_CSV = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main-nn/0827result/maps_features_with_complexity.csv"

# 3) 训练超参（全局默认下限；各子阶段会覆盖）
N_STAGES_BASE = 5                       # 先分 5 桶；再把 2/3/4 分半为 a/b
MIN_PER_STAGE = 10
MIN_EPISODES_PER_STAGE = 300
THRESHOLD = 0.80
WINDOW_SIZE = 120
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 32000
DECAY_RANGE = 10_000
MAX_EPISODE_SECONDS = 70
LOG_DIR = "logs"
RUN_DIR = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main-nn/final_training_refined"
MODEL_OUT = "models/best_model.pt"

# 防卡死：目标智能体连续多少步位置不变就提前终止该集（0=关闭）
STUCK_PATIENCE = 0
# ============================================================

# ============ 项目根路径（确保能 import g2rl.*） ============
project_root = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main-nn"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============ 项目模块 ============
from g2rl.environment import G2RLEnv
from g2rl.agent import DDQNAgent
from g2rl.network import CRNNModel

# ============ 可选：Pogema A*（没有就用内置贪心） ============
try:
    from pogema import AStarAgent as _PogemaAStarAgent
    HAVE_POGEMA = True
except Exception:
    HAVE_POGEMA = False


# ================= 公用工具 =================
def _pwrite(pbar, msg: str):
    """安全写进度条；pbar 可能没有 total 或为 None。"""
    try:
        if pbar is not None and hasattr(pbar, "write"):
            pbar.write(msg); return
    except Exception:
        pass
    print(msg)


def _find_array(obj):
    import numpy as np
    try:
        import torch
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
        preferred = ("view_cache","obs","observation","state","tensor","grid","image","local_obs","global_obs")
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


def _obs_to_tensor_CDHW(s, device, expected_c: int):
    """把任何观测统一到 [1, C, D, H, W] 并对齐通道数。"""
    import numpy as np, torch
    arr = _find_array(s)
    if arr is None:
        raise ValueError("无法从观测中提取数组/张量。")
    arr = np.array(arr)

    # 归一到 [C, D, H, W]
    if arr.ndim == 5:
        if arr.shape[1] == expected_c: arr = arr[0]
        elif arr.shape[-1] == expected_c: arr = np.transpose(arr, (0,4,1,2,3))[0]
        else: arr = arr[0]
    elif arr.ndim == 4:
        if arr.shape[0] == expected_c: pass
        elif arr.shape[-1] == expected_c: arr = np.transpose(arr, (3,0,1,2))
        elif arr.shape[1] == expected_c: arr = np.transpose(arr, (1,0,2,3))
    elif arr.ndim == 3:
        arr = arr[None, ...]
    elif arr.ndim == 2:
        arr = arr[None, None, ...]
    else:
        raise ValueError(f"观测维度不支持：shape={arr.shape}, ndim={arr.ndim}")

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
    return torch.tensor(arr_fixed[None, ...], dtype=torch.float32, device=device)


def _pogema_num_agents_from_obs(obs) -> int:
    try:
        return int(len(obs))
    except Exception:
        return 1


def _idle_action_index(env, default_idx=0) -> int:
    if hasattr(env, "actions") and isinstance(env.actions, (list, tuple)):
        for i, name in enumerate(env.actions):
            if isinstance(name, str) and name.lower() in ("idle", "stay", "noop", "wait"):
                return i
    return default_idx


def _safe_get_action_space(env):
    asp = getattr(env, "action_space", None)
    if asp is not None and hasattr(asp, "n"):
        return asp
    gas = getattr(env, "get_action_space", None)
    if callable(gas):
        try:
            out = gas()
            if hasattr(out, "n"): return out
            if isinstance(out, (list, tuple)): return SimpleNamespace(n=len(out))
            if isinstance(out, int) and out > 0: return SimpleNamespace(n=out)
        except Exception:
            pass
    if hasattr(env, "actions"):
        try:
            return SimpleNamespace(n=len(env.actions))
        except Exception:
            pass
    try: env.reset()
    except Exception: pass
    asp = getattr(env, "action_space", None)
    if asp is not None and hasattr(asp, "n"): return asp
    gas = getattr(env, "get_action_space", None)
    if callable(gas):
        try:
            out = gas()
            if hasattr(out, "n"): return out
            if isinstance(out, (list, tuple)): return SimpleNamespace(n=len(out))
            if isinstance(out, int) and out > 0: return SimpleNamespace(n=out)
        except Exception:
            pass
    if hasattr(env, "actions"):
        try:
            return SimpleNamespace(n=len(env.actions))
        except Exception:
            pass
    return SimpleNamespace(n=5)


def build_env_from_raw(raw_cfg: dict) -> G2RLEnv:
    """安全构建 env（只传 __init__ 支持的参数），并挂 grid/starts/goals 属性便于规则判断。"""
    sig = inspect.signature(G2RLEnv.__init__)
    allowed = {
        p.name for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    allowed.discard("self")
    ctor_cfg = {k: v for k, v in raw_cfg.items() if k in allowed}
    env = G2RLEnv(**ctor_cfg)

    # 辅助属性
    try:
        import numpy as _np
        if "grid" in raw_cfg:
            env.grid = (_np.array(raw_cfg["grid"]) > 0).astype(_np.uint8)
    except Exception:
        env.grid = getattr(env, "grid", None)
    if "starts" in raw_cfg: env.starts = raw_cfg["starts"]
    if "goals"  in raw_cfg: env.goals  = raw_cfg["goals"]
    return env


# ================== 复杂度合并 ==================
def _merge_complexity_from_csv(base_map_settings: Dict[str, dict]) -> Dict[str, dict]:
    if not os.path.exists(COMPLEXITY_CSV):
        raise FileNotFoundError(f"未找到复杂度CSV：{COMPLEXITY_CSV}")

    df = pd.read_csv(COMPLEXITY_CSV)

    if "nn_complexity" not in df.columns:
        if "nn_pred_success_rate" in df.columns:
            df["nn_complexity"] = 1.0 - pd.to_numeric(df["nn_pred_success_rate"], errors="coerce")
        else:
            raise ValueError("复杂度CSV缺少 nn_complexity 且没有 nn_pred_success_rate 无法推导。")

    df["nn_complexity"] = pd.to_numeric(df["nn_complexity"], errors="coerce")

    # 直接ID匹配或复合键
    csv_id_candidates  = [c for c in ["config_id", "grid_hash", "name", "map_id"] if c in df.columns]
    yaml_id_candidates = set()
    for spec in base_map_settings.values():
        yaml_id_candidates.update(spec.keys())
    yaml_id_candidates = [c for c in ["config_id", "grid_hash", "name", "map_id"] if c in yaml_id_candidates]
    common_ids = [c for c in csv_id_candidates if c in yaml_id_candidates]
    chosen_id = common_ids[0] if common_ids else None
    use_composite = (chosen_id is None)

    if not use_composite:
        agg = (
            df.groupby(chosen_id, dropna=False)["nn_complexity"]
              .mean().reset_index().rename(columns={"nn_complexity": "complexity"})
        )
        comp_map = dict(zip(agg[chosen_id].astype(str), agg["complexity"].astype(float)))
    else:
        needed = ["size", "num_agents", "density", "obs_radius", "max_episode_steps"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"无法用复合键匹配，CSV 缺少列：{missing}")

        dfx = df.copy()
        dfx["_size"] = pd.to_numeric(dfx["size"], errors="coerce").astype("Int64")
        dfx["_nag"]  = pd.to_numeric(dfx["num_agents"], errors="coerce").astype("Int64")
        dfx["_den"]  = pd.to_numeric(dfx["density"], errors="coerce").round(4)
        dfx["_obs"]  = pd.to_numeric(dfx["obs_radius"], errors="coerce").astype("Int64")
        dfx["_mep"]  = pd.to_numeric(dfx["max_episode_steps"], errors="coerce").astype("Int64")
        comp = (
            dfx.dropna(subset=["_size","_nag","_den","_obs","_mep"])
               .groupby(["_size","_nag","_den","_obs","_mep"])["nn_complexity"]
               .mean().reset_index().rename(columns={"nn_complexity":"complexity"})
        )
        comp_map = { (int(r["_size"]), int(r["_nag"]), float(r["_den"]), int(r["_obs"]), int(r["_mep"])): float(r["complexity"])
                     for _, r in comp.iterrows() }

    matched, unmatched = 0, 0
    out: Dict[str, dict] = {}

    for name, spec in base_map_settings.items():
        new_spec = dict(spec)
        cpx_val = np.nan

        if not use_composite:
            key = spec.get(chosen_id, None)
            if key is None and chosen_id == "name":
                key = name
            if key is not None:
                cpx_val = comp_map.get(str(key), np.nan)
        else:
            try:
                size  = int(spec.get("size"))
                nag   = int(spec.get("num_agents"))
                den   = float(spec.get("density"))
                obs   = int(spec.get("obs_radius"))
                mep   = int(spec.get("max_episode_steps"))
                tup   = (size, nag, round(den,4), obs, mep)
                cpx_val = comp_map.get(tup, np.nan)
            except Exception:
                cpx_val = np.nan

        if np.isfinite(cpx_val):
            matched += 1
            new_spec["complexity"] = float(cpx_val)
        else:
            unmatched += 1
            new_spec["complexity"] = np.nan

        out[name] = new_spec

    print(f"🧩 复杂度匹配统计：matched={matched}, unmatched={unmatched}")
    if unmatched > 0:
        examples = [k for k,v in out.items() if not np.isfinite(v.get('complexity', np.nan))][:5]
        print(f"⚠️ 有 {unmatched} 张地图未匹配到复杂度（未参与量化）。示例：{examples}")
    return out


# ================== 分位 + 2/3/4 切半为 a/b ==================
def _build_stages_refined(df: pd.DataFrame) -> List[dict]:
    """
    先按复杂度分 5 桶（0..4）；再把 2/3/4 桶按中位数各自切半：
      0, 1, 2a, 2b, 3a, 3b, 4a, 4b
    """
    dfn = df.dropna(subset=["complexity"]).copy()
    if len(dfn) == 0:
        raise ValueError("没有可用地图用于 complexity 课程（complexity 全 NaN）。")

    qs = np.linspace(0, 1, N_STAGES_BASE + 1)  # 6 个边界点
    edges = np.quantile(dfn["complexity"].values, qs)  # len=6

    buckets = []
    for i in range(N_STAGES_BASE):
        lo, hi = float(edges[i]), float(edges[i+1]) + 1e-12
        sub = dfn[(dfn["complexity"] >= lo) & (dfn["complexity"] < hi)].copy()
        buckets.append((i, lo, hi, sub))

    stages = []

    # stage 0 / 1 原样
    for i in [0, 1]:
        st = buckets[i]
        stages.append({"name": str(i), "cpx_min": st[1], "cpx_max": st[2], "items": st[3].to_dict("records")})

    def split_half(name_a, name_b, subdf):
        sub = subdf.sort_values("complexity").reset_index(drop=True)
        if len(sub) <= 1:
            return (
                {"name": name_a, "cpx_min": float(sub["complexity"].min() if len(sub) else 0.0),
                 "cpx_max": float(sub["complexity"].max() if len(sub) else 0.0),
                 "items": sub.to_dict("records")},
                {"name": name_b, "cpx_min": float(sub["complexity"].min() if len(sub) else 0.0),
                 "cpx_max": float(sub["complexity"].max() if len(sub) else 0.0),
                 "items": []},
            )
        mid = max(1, len(sub) // 2)
        a = sub.iloc[:mid]
        b = sub.iloc[mid:]
        return (
            {"name": name_a, "cpx_min": float(a["complexity"].min()), "cpx_max": float(a["complexity"].max()), "items": a.to_dict("records")},
            {"name": name_b, "cpx_min": float(b["complexity"].min()), "cpx_max": float(b["complexity"].max()), "items": b.to_dict("records")},
        )

    # 2 → 2a/2b
    s2 = buckets[2]
    s2a, s2b = split_half("2a", "2b", s2[3])
    stages += [s2a, s2b]

    # 3 → 3a/3b
    s3 = buckets[3]
    s3a, s3b = split_half("3a", "3b", s3[3])
    stages += [s3a, s3b]

    # 4 → 4a/4b
    s4 = buckets[4]
    s4a, s4b = split_half("4a", "4b", s4[3])
    stages += [s4a, s4b]

    return stages


# ================== 动作无效 & 冲突预测 ==================
def _invalid_actions_for(env, obs_i, idle_idx=0) -> Tuple[List[int], Dict[int, Tuple[int,int]]]:
    invalid = []
    try:
        x, y = obs_i['global_xy']
    except Exception:
        return invalid, {idle_idx: (0,0)}

    moves = {
        "up":    (-1, 0),
        "right": (0, 1),
        "down":  (1, 0),
        "left":  (0, -1),
    }
    mapping = {}
    if hasattr(env, "actions") and isinstance(env.actions, (list, tuple)):
        for idx, name in enumerate(env.actions):
            if isinstance(name, str):
                low = name.lower()
                if low in ("idle", "stay", "noop", "wait"):
                    mapping[idx] = (0, 0)
                elif "up" in low:
                    mapping[idx] = moves["up"]
                elif "right" in low:
                    mapping[idx] = moves["right"]
                elif "down" in low:
                    mapping[idx] = moves["down"]
                elif "left" in low:
                    mapping[idx] = moves["left"]
    if not mapping:
        mapping = {idle_idx:(0,0), 1:(-1,0), 2:(0,1), 3:(1,0), 4:(0,-1)}

    grid = getattr(env, "grid", None)
    H, W = (grid.shape[:2] if grid is not None else (None, None))
    for a, (dx, dy) in mapping.items():
        nx, ny = x + dx, y + dy
        if H is not None and (nx < 0 or ny < 0 or nx >= H or ny >= W):
            invalid.append(a); continue
        if grid is not None:
            try:
                if grid[nx, ny] != 0:
                    invalid.append(a); continue
            except Exception:
                pass
    return list(set(invalid)), mapping


def opposite_of(prev_action: int, idx2move: dict, idle_idx: int) -> set:
    v = idx2move.get(prev_action, None)
    if v is None or v == (0, 0):
        return set()
    inv = (-v[0], -v[1])
    return {a for a, mv in idx2move.items() if mv == inv}


def will_conflict_next_step(obs_prev, actions, target_idx: int, idx2move: dict) -> bool:
    """预测下一步顶点/边冲突。"""
    try:
        n = len(obs_prev)
        cur = [tuple(obs_prev[i]['global_xy']) for i in range(n)]
    except Exception:
        return False

    nxt = []
    for i in range(len(actions)):
        a = actions[i]
        dx, dy = idx2move.get(a, (0, 0))
        x, y = cur[i]
        nxt.append((x + dx, y + dy))

    tgt_next = nxt[target_idx]
    if any((i != target_idx and nxt[i] == tgt_next) for i in range(len(nxt))):
        return True

    tgt_cur = cur[target_idx]
    for i in range(len(nxt)):
        if i == target_idx:
            continue
        if nxt[i] == tgt_cur and nxt[target_idx] == cur[i]:
            return True

    return False


# ================== 简单贪心（用于队友） ==================
class GreedyAgent:
    def __init__(self, env, idle_idx: int, idx_to_move: Dict[int, tuple]):
        self.env = env
        self.idle_idx = idle_idx
        self.idx_to_move = idx_to_move

    def act(self, obs_i, goal_xy):
        try:
            x, y = obs_i['global_xy']
            gx, gy = goal_xy
        except Exception:
            return self.idle_idx

        best_a, best_gain = self.idle_idx, 0
        d0 = abs(x - gx) + abs(y - gy)
        grid = getattr(self.env, "grid", None)
        H, W = (grid.shape[:2] if grid is not None else (None, None))

        for a, (dx, dy) in self.idx_to_move.items():
            nx, ny = x + dx, y + dy
            if H is not None and (nx < 0 or ny < 0 or nx >= H or ny >= W):
                continue
            if grid is not None:
                try:
                    if grid[nx, ny] != 0:
                        continue
                except Exception:
                    pass
            d1 = abs(nx - gx) + abs(ny - gy)
            gain = d0 - d1
            if gain > best_gain:
                best_gain, best_a = gain, a
        return best_a


# ================== Scheduler ==================
class ComplexityScheduler:
    """
    - 先用 5 桶分位，然后把 2/3/4 各自再切半：0,1,2a,2b,3a,3b,4a,4b
    - 每阶段至少跑 min_episodes_per_stage 集，达标阈值后晋级
    - 同图冷却/连败回退/黑名单（带保底）/兜底轮换（≤8 agents 优先，避开 cpx≈0.102）
    """

    # 同图保护
    MAX_CONSEC_PER_MAP = 5
    FAIL_STREAK_BAN    = 5
    COOLDOWN_EPS       = 200

    def __init__(self,
                 base_map_settings: Dict[str, dict],
                 min_per_stage: int = 10,
                 min_episodes_per_stage: int = 300,
                 threshold: float = 0.80,
                 window_size: int = 120,
                 shuffle_each_stage: bool = True,
                 seed: int = 0):

        self.min_episodes_per_stage = int(min_episodes_per_stage)
        self.threshold = float(threshold)
        self.window_size = int(window_size)

        rows = [{"name": n, "complexity": s.get("complexity", np.nan), "spec": s}
                for n, s in base_map_settings.items()]
        df = pd.DataFrame(rows)

        stages = _build_stages_refined(df)

        self._rng = random.Random(seed)
        self._stage_items: List[List[dict]] = []
        self._stage_edges: List[Tuple[float,float]] = []
        self._stage_names: List[str] = []

        for st in stages:
            items = list(st["items"])
            if shuffle_each_stage:
                self._rng.shuffle(items)
            # 保底数量（若某桶太小，用全局最易补足）
            if len(items) < min_per_stage:
                all_sorted = df.dropna(subset=["complexity"]).sort_values("complexity").to_dict("records")
                need = min_per_stage - len(items)
                items += all_sorted[:need]
            self._stage_items.append(items)
            self._stage_edges.append((st["cpx_min"], st["cpx_max"]))
            self._stage_names.append(str(st["name"]))

        self.current_stage = 0
        self.max_stage = len(self._stage_items) - 1

        # 统计
        self._idx_in_stage = 0
        self._win = deque(maxlen=self.window_size)
        self._ep_in_stage = 0
        self._succ_in_stage = 0
        self._repeat_fail = 0
        self._shrink_applied = set()

        # 指针/连败
        nS = self.max_stage + 1
        self._pos = [0] * nS
        self._sorted = [False] * nS
        self._succ_streak = [0] * nS
        self._fail_streak = [0] * nS

        # 冷却/黑名单
        self._map_fail_streak = defaultdict(int)
        self._map_fail_total  = defaultdict(int)
        self._cooldown_until  = dict()             # ((stage,name)->到第几集)
        self._current_item_name = None
        self._map_state = defaultdict(lambda: {"consec": 0, "fail_streak": 0, "cooldown_until": -1})
        self._global_ep = 0

        self._ban_after = 18
        self._banned = set()
        self._ban_min_left = 3
        self._ban_soft_stage_idx = 1               # stage<=1 不拉黑（只冷却）
        self._long_cooldown = 300

        # 子阶段 agents 上限
        self._agents_cap_by_stage = {
            "2a": 4, "2b": 8, "3a": 8, "3b": 12, "4a": 12, "4b": 16
        }

        # 避坑带
        self._cpx_trap_center = 0.102
        self._cpx_trap_eps = 0.008

        # 兜底轮换下标
        self._easy_rotate = 0

    def current_stage_name(self) -> str:
        return self._stage_names[self.current_stage]

    # ---------- 候选过滤 ----------
    def _eligible_pool(self, candidates, ep_idx, pbar=None):
        pool = []
        s_idx = self.current_stage
        stage_name = self._stage_names[s_idx]
        agents_cap = self._agents_cap_by_stage.get(stage_name, None)

        for r in candidates:
            name = r["name"]
            cd_until = self._cooldown_until.get((s_idx, name), -1)
            if ep_idx < cd_until:
                continue
            if (s_idx, name) in self._banned:
                continue
            # 同图连续保护（全局）
            s = self._map_state[name]
            if self._global_ep < s["cooldown_until"]:
                continue
            if s["consec"] >= self.MAX_CONSEC_PER_MAP:
                s["cooldown_until"] = self._global_ep + self.COOLDOWN_EPS
                s["consec"] = 0
                continue

            # 子阶段 agents 上限
            try:
                na = int(r["spec"].get("num_agents", r["spec"].get("agents", 99)))
                if agents_cap is not None and na > agents_cap:
                    continue
            except Exception:
                pass

            # 尽量避开 cpx≈0.102（池够大时才过滤）
            try:
                cpx = r["spec"].get("complexity", None)
                if cpx is not None and np.isfinite(cpx):
                    if abs(float(cpx) - self._cpx_trap_center) <= self._cpx_trap_eps:
                        continue
            except Exception:
                pass

            pool.append(r)

        # 若全冷却/全黑/全超限，兜底：优先 ≤8 agents 且避开 cpx≈0.102，轮换最容易前K（<=8）
        if not pool:
            easy_sorted = sorted(
                [x for x in candidates if (s_idx, x["name"]) not in self._banned],
                key=lambda z: (0 if np.isfinite(z["spec"].get("complexity", np.nan)) else 1,
                               float(z["spec"].get("complexity", float("inf"))))
            )
            easy_small = [x for x in easy_sorted if int(x["spec"].get("num_agents", 99)) <= 8]
            bucket = easy_small if easy_small else easy_sorted
            bucket = [x for x in bucket if not (x["spec"].get("complexity") and abs(float(x["spec"]["complexity"]) - self._cpx_trap_center) <= self._cpx_trap_eps)]
            if not bucket:
                bucket = easy_sorted

            headK = min(8, max(1, len(bucket)))
            head = bucket[:headK]
            if head:
                pick = head[self._easy_rotate % len(head)]
                self._easy_rotate += 1
                _pwrite(pbar, f"🧯 Stage {stage_name} 兜底：候选均在冷却，轮换最容易前{headK} → {pick['name']}")
                pool = [pick]
        return pool

    def get_updated_map_settings(self, pbar=None) -> Dict[str, dict]:
        if self.current_stage > self.max_stage:
            return {}

        s_idx = self.current_stage
        items = self._stage_items[s_idx]
        if not items:
            raise RuntimeError(f"Stage {self._stage_names[s_idx]} 没有地图。")

        # 首次排序：按 complexity 升序（NaN 放后）
        if not self._sorted[s_idx]:
            def _key(r):
                c = r["spec"].get("complexity", None)
                return (1, float("inf")) if (c is None or not np.isfinite(c)) else (0, float(c))
            items = sorted(items, key=_key)
            self._stage_items[s_idx] = items
            self._pos[s_idx] = 0
            self._sorted[s_idx] = True

        ep_idx = self._ep_in_stage
        usable = self._eligible_pool(items, ep_idx, pbar=pbar)

        if not usable:
            # 真·全黑：解封最易一张
            easy = min(items, key=lambda r: r["spec"].get("complexity", float("inf")))
            self._banned.discard((s_idx, easy["name"]))
            usable = [easy]
            _pwrite(pbar, f"🛟 Stage {self._stage_names[s_idx]} 黑名单过多，解封最容易地图：{easy['name']}")

        idx = max(0, min(self._pos[s_idx], len(usable) - 1))
        item = usable[idx]

        # 特判：cpx≈0.102 → 若可选同池其他更安全项就替换
        try:
            cpx_here = item["spec"].get("complexity", None)
            if (cpx_here is not None and np.isfinite(cpx_here) and abs(float(cpx_here) - self._cpx_trap_center) <= self._cpx_trap_eps):
                safer = [r for r in usable
                         if r["name"] != item["name"]
                         and not (r["spec"].get("complexity") and abs(float(r["spec"]["complexity"]) - self._cpx_trap_center) <= self._cpx_trap_eps)]
                if safer:
                    item = safer[idx % len(safer)]
        except Exception:
            pass

        self._current_item_name = item["name"]
        return {item["name"]: item["spec"]}

    def add_episode_result(self, success: int, pbar=None):
        s_val = 1 if success else 0
        self._win.append(s_val)
        self._ep_in_stage += 1
        self._succ_in_stage += s_val

        s_idx = self.current_stage
        L = len(self._stage_items[s_idx])
        self._pos[s_idx] = max(0, min(self._pos[s_idx], max(L - 1, 0)))

        if success:
            self._succ_streak[s_idx] += 1
            self._fail_streak[s_idx] = 0
            if self._succ_streak[s_idx] >= 1:
                self._pos[s_idx] = min(self._pos[s_idx] + 1, L - 1)
                self._succ_streak[s_idx] = 0
        else:
            self._fail_streak[s_idx] += 1
            self._succ_streak[s_idx] = 0
            if self._fail_streak[s_idx] >= 1:
                self._pos[s_idx] = max(self._pos[s_idx] - 1, 0)
                self._fail_streak[s_idx] = 0

        name = self._current_item_name
        if name is None:
            self._global_ep += 1
            return
        key = (s_idx, name)

        st = self._map_state[name]
        st["consec"] += 1
        if success:
            st["fail_streak"] = 0
        else:
            st["fail_streak"] += 1
            if st["fail_streak"] >= self.FAIL_STREAK_BAN:
                st["cooldown_until"] = self._global_ep + self.COOLDOWN_EPS
                st["consec"] = 0

        # stage 层面的连败统计 & 特判 cpx≈0.102
        try:
            cur = next((r for r in self._stage_items[s_idx] if r["name"] == name), None)
            cpx = cur["spec"].get("complexity", None) if cur else None
        except Exception:
            cpx = None

        if (not success) and (cpx is not None) and np.isfinite(cpx) and abs(float(cpx) - self._cpx_trap_center) <= self._cpx_trap_eps:
            self._pos[s_idx] = max(self._pos[s_idx] - 2, 0)
            self._cooldown_until[key] = self._ep_in_stage + 180
            _pwrite(pbar, f"⛔ cpx≈0.102 快速退避：Stage {self._stage_names[s_idx]} 回退2格并冷却 {name}")
            self._global_ep += 1
            return

        # 连败回退 + 冷却
        if not success:
            self._map_fail_streak[key] += 1
            self._map_fail_total[key]  += 1

        if (not success) and self._map_fail_streak[key] >= 3:
            self._pos[s_idx] = max(self._pos[s_idx] - 1, 0)
            self._cooldown_until[key] = self._ep_in_stage + 120
            self._map_fail_streak[key] = 0
            _pwrite(pbar, f"📉 连败触发：Stage {self._stage_names[s_idx]} 回退一格并冷却 {name}")

        # 拉黑（带保底：低阶不拉黑 / 剩余太少不拉黑）
        if self._map_fail_total[key] >= self._ban_after:
            if s_idx <= self._ban_soft_stage_idx:
                self._cooldown_until[key] = self._ep_in_stage + self._long_cooldown
                self._map_fail_streak[key] = 0
                self._map_fail_total[key]  = 0
                _pwrite(pbar, f"🧊 保护：低阶段不拉黑，冷却 {name}（{self._long_cooldown} ep）")
            else:
                remain = sum((s_idx, r["name"]) not in self._banned for r in self._stage_items[s_idx])
                if remain <= self._ban_min_left:
                    self._cooldown_until[key] = self._ep_in_stage + self._long_cooldown
                    self._map_fail_streak[key] = 0
                    self._map_fail_total[key]  = 0
                    if self._pos[s_idx] > 0:
                        self._pos[s_idx] -= 1
                    _pwrite(pbar, f"🛡️ 保护：Stage {self._stage_names[s_idx]} 可用≤{self._ban_min_left}，不拉黑 {name}，改冷却 {self._long_cooldown} ep")
                else:
                    self._banned.add(key)
                    if self._pos[s_idx] > 0:
                        self._pos[s_idx] -= 1
                    self._map_fail_streak[key] = 0
                    self._map_fail_total[key]  = 0
                    _pwrite(pbar, f"🚫 Stage {self._stage_names[s_idx]} 拉黑地图：{name}（累计失败≥{self._ban_after}）")

        self._global_ep += 1

    def window_sr(self) -> float:
        return float(sum(self._win) / len(self._win)) if len(self._win) else 0.0

    def stage_sr(self) -> float:
        return float(self._succ_in_stage / max(1, self._ep_in_stage))

    def should_advance(self) -> bool:
        if self._ep_in_stage < self.min_episodes_per_stage:
            return False
        return self.stage_sr() >= self.threshold

    def advance(self, pbar=None):
        lo, hi = self._stage_edges[self.current_stage]
        _pwrite(pbar, f"✅ 通过 Stage {self.current_stage_name()} | SR(win)={self.window_sr():.2f} / SR(stage)={self.stage_sr():.2f} | 区间[{lo:.4f}, {hi:.4f}] → 下一阶段")
        self.current_stage += 1
        self._reset_stage_stats()
        self._repeat_fail = 0

    def repeat_stage(self, pbar=None):
        self._repeat_fail += 1
        _pwrite(pbar, f"🔁 未达标，重复 Stage {self.current_stage_name()}（已训练 {self._ep_in_stage} ep，SR={self.stage_sr():.2f}，连续失败={self._repeat_fail}）")
        # 连续失败≥2 → 缩池（保留更易前 1/3）
        s_idx = self.current_stage
        if self._repeat_fail >= 2 and s_idx not in self._shrink_applied:
            items = self._stage_items[s_idx]
            if items:
                def _cmp_key(r):
                    c = r['spec'].get('complexity', None)
                    return (1, 1e9) if (c is None or not np.isfinite(c)) else (0, float(c))
                items_sorted = sorted(items, key=_cmp_key)
                keep = max(5, len(items_sorted) // 3)
                new_pool = items_sorted[:keep]
                self._stage_items[s_idx] = new_pool
                self._shrink_applied.add(s_idx)
                lo, hi = self._stage_edges[s_idx]
                _pwrite(pbar, f"📉 降难：Stage {self.current_stage_name()} 地图池由 {len(items)} → {len(new_pool)}（区间≈[{lo:.4f},{hi:.4f}]）")
                self._idx_in_stage = 0
        self._reset_stage_stats()

    def _reset_stage_stats(self):
        self._idx_in_stage = 0
        self._win.clear()
        self._ep_in_stage = 0
        self._succ_in_stage = 0
        if 0 <= self.current_stage <= self.max_stage:
            s = self.current_stage
            self._pos[s] = 0
            self._succ_streak[s] = 0
            self._fail_streak[s] = 0
            self._sorted[s] = False

    def is_done(self) -> bool:
        return self.current_stage > self.max_stage


# ================== 训练 ==================
def get_timestamp() -> str:
    return datetime.now().strftime('%H-%M-%d-%m-%Y')


def _rb_is_iterable(rb) -> bool:
    try:
        iter(rb)
        return True
    except TypeError:
        return False
    except Exception:
        return False


def train(
    model: torch.nn.Module,
    map_settings: Dict[str, dict],
    map_probs: Union[List[float], None],
    num_episodes: int = 300,
    batch_size: int = 128,
    decay_range: int = 1000,
    log_dir='logs',
    lr: float = 0.001,
    replay_buffer_size: int = 1000,
    device: str = 'cuda',
    scheduler: Optional[ComplexityScheduler] = None,
    max_episode_seconds: int = 60,
    run_dir: Optional[str] = None
) -> DDQNAgent:

    episode = 0
    success_count_total = 0

    timestamp = get_timestamp()
    run_dir = Path(log_dir) / timestamp if run_dir is None else Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    # —— 子阶段超参覆盖（按阶段名）
    STAGE_OVERRIDES_BY_NAME = {
        "0":  {"max_episode_seconds": 45, "batch_size": 48,  "replay_buffer_size":  8000, "target_tau": 0.02, "eps_min": 0.10},
        "1":  {"max_episode_seconds": 60, "batch_size": 64,  "replay_buffer_size": 16000, "target_tau": 0.02, "eps_min": 0.15},

        "2a": {"max_episode_seconds": 95, "batch_size": 96,  "replay_buffer_size": 32000, "target_tau": 0.05,  "eps_min": 0.35,
               "replay_mix": (0.65, 0.30, 0.05), "risk_weight": 0.30, "risk_clip": 2.0},
        "2b": {"max_episode_seconds": 90, "batch_size": 96,  "replay_buffer_size": 32000, "target_tau": 0.04,  "eps_min": 0.28,
               "replay_mix": (0.55, 0.35, 0.10), "risk_weight": 0.25, "risk_clip": 2.0},

        "3a": {"max_episode_seconds": 95, "batch_size": 112, "replay_buffer_size": 40000, "target_tau": 0.04,  "eps_min": 0.28,
               "replay_mix": (0.50, 0.40, 0.10), "risk_weight": 0.22, "risk_clip": 2.5},
        "3b": {"max_episode_seconds": 90, "batch_size": 112, "replay_buffer_size": 45000, "target_tau": 0.035, "eps_min": 0.22,
               "replay_mix": (0.40, 0.45, 0.15), "risk_weight": 0.18, "risk_clip": 3.0},

        "4a": {"max_episode_seconds": 95, "batch_size": 128, "replay_buffer_size": 50000, "target_tau": 0.035, "eps_min": 0.20,
               "replay_mix": (0.35, 0.50, 0.15), "risk_weight": 0.15, "risk_clip": 3.0},
        "4b": {"max_episode_seconds": 95, "batch_size": 128, "replay_buffer_size": 60000, "target_tau": 0.03,  "eps_min": 0.16,
               "replay_mix": (0.30, 0.55, 0.15), "risk_weight": 0.12, "risk_clip": 3.0},
    }

    # —— 初始化第一个 env & 动作空间
    first_name = next(iter(map_settings))
    first_env = build_env_from_raw(map_settings[first_name])
    try:
        first_env.reset()
    except Exception:
        pass

    try:
        asp = _safe_get_action_space(first_env)
        n_actions = int(getattr(asp, "n", 5))
    except Exception:
        n_actions = 5
    action_space_list = list(range(n_actions))
    print(f"✅ n_actions = {n_actions}")

    # 预热模型
    try:
        obs0, _ = first_env.reset()
    except Exception:
        obs0 = first_env.reset()
    state0 = obs0[0] if isinstance(obs0, (list, tuple)) else obs0
    in_channels = 11
    with torch.no_grad():
        _ = model(_obs_to_tensor_CDHW(state0, device, expected_c=in_channels))
    print(f"🔥 warmed model with in_channels={in_channels}")

    def preprocess_single(s):
        return _obs_to_tensor_CDHW(s, device, expected_c=in_channels).squeeze(0)

    # 构造 Agent
    agent = DDQNAgent(
        model, model, action_space_list,
        lr=lr, decay_range=decay_range, device=device,
        replay_buffer_size=replay_buffer_size, obs_preprocessor=preprocess_single
    )
    if not hasattr(agent, "success_buffer"):
        agent.success_buffer = deque(maxlen=20000)

    training_logs = []
    pbar = tqdm(desc='Episodes', unit='ep', dynamic_ncols=True, total=None)

    while True:
        if scheduler is None:
            raise ValueError("需要 scheduler 才能按阶段课程训练直到全部完成。")
        if scheduler.is_done():
            break

        # 1) 取当前阶段 & 地图
        cur_map_cfg = scheduler.get_updated_map_settings(pbar=pbar)
        map_name, cfg = next(iter(cur_map_cfg.items()))
        env = build_env_from_raw(cfg)

        stage_name = scheduler.current_stage_name()
        lo, hi = scheduler._stage_edges[scheduler.current_stage]
        cpx_val = cfg.get("complexity", None)
        na = int(cfg.get("num_agents", getattr(env, "num_agents", -1)))
        if cpx_val is not None and np.isfinite(cpx_val):
            _pwrite(pbar, f"🟢 使用地图：{map_name} | Stage {stage_name} | Agents={na} | Complexity={float(cpx_val):.3f}")
        else:
            _pwrite(pbar, f"🟢 使用地图：{map_name} | Stage {stage_name} | Agents={na}")

        # 2) reset
        try:
            obs, info = env.reset()
        except Exception:
            obs = env.reset()
            info = {}

        # 3) 阶段覆盖
        ovr = STAGE_OVERRIDES_BY_NAME.get(stage_name, {})
        used_batch_size  = int(ovr.get("batch_size", batch_size))
        used_max_seconds = int(ovr.get("max_episode_seconds", max_episode_seconds))
        target_tau       = float(ovr.get("target_tau", 0.02))
        eps_min          = float(ovr.get("eps_min", 0.1))
        replay_mix       = ovr.get("replay_mix", None)
        risk_weight      = float(ovr.get("risk_weight", 0.0))
        risk_clip        = float(ovr.get("risk_clip", 2.0))

        # 动态 LR（高阶段略升）
        if hasattr(agent, "optimizer"):
            if not hasattr(agent, "_base_lrs"):
                agent._base_lrs = [pg["lr"] for pg in agent.optimizer.param_groups]
                agent._lr_mode = "base"
            if stage_name in ("3a","3b","4a","4b") and agent._lr_mode != "high":
                for g, lr0 in zip(agent.optimizer.param_groups, agent._base_lrs):
                    g["lr"] = lr0 * 1.2
                agent._lr_mode = "high"
                _pwrite(pbar, f"🚀 高阶段LR ×1.2 → {[pg['lr'] for pg in agent.optimizer.param_groups]}")
            elif stage_name in ("2a","2b") and agent._lr_mode != "mid":
                for g, lr0 in zip(agent.optimizer.param_groups, agent._base_lrs):
                    g["lr"] = lr0 * 1.1
                agent._lr_mode = "mid"

        # 调 tau
        for attr in ["tau","target_tau","soft_update_tau","target_update_tau"]:
            if hasattr(agent, attr):
                setattr(agent, attr, target_tau)
                break

        # 动态扩容经验池
        if "replay_buffer_size" in ovr:
            new_cap = int(ovr["replay_buffer_size"])
            try:
                if hasattr(agent, "resize_buffer") and callable(agent.resize_buffer):
                    agent.resize_buffer(new_cap)
                elif hasattr(agent, "replay_buffer") and hasattr(agent.replay_buffer, "maxlen"):
                    old_buf = agent.replay_buffer
                    tmp = list(old_buf)
                    agent.replay_buffer = deque(tmp, maxlen=new_cap)
                elif hasattr(agent, "replay_buffer") and hasattr(agent.replay_buffer, "capacity"):
                    agent.replay_buffer.capacity = new_cap
                _pwrite(pbar, f"🗃️ Stage {stage_name}: replay buffer → {new_cap}")
            except Exception as e:
                _pwrite(pbar, f"⚠️ 扩容经验池失败：{e}")

        # epsilon 最低值（阶段回弹）
        if hasattr(agent, "epsilon"):
            agent.epsilon = max(eps_min, float(getattr(agent, "epsilon", 0.1)))

        # 观测长度作为代理数
        n_agents_pg = _pogema_num_agents_from_obs(obs)
        idle_idx = _idle_action_index(env, 0)
        invalid_0, idx2move = _invalid_actions_for(env, obs[0], idle_idx)

        # 选目标 agent：前期最短距；中高阶段取中位
        try:
            dists = [(abs(obs[i]['global_xy'][0] - env.goals[i][0]) +
                      abs(obs[i]['global_xy'][1] - env.goals[i][1]), i)
                     for i in range(n_agents_pg)]
            if scheduler._ep_in_stage < 100:
                target_idx = min(dists)[1]
            elif stage_name in ("2a","2b","3a","3b","4a","4b"):
                target_idx = sorted(dists)[len(dists)//2][1]
            else:
                target_idx = min(dists)[1]
        except Exception:
            target_idx = int(np.random.randint(n_agents_pg))

        # 队友：pogema A* or 内置 Greedy
        teammates = [None] * n_agents_pg
        for i in range(n_agents_pg):
            if i == target_idx:
                continue
            if HAVE_POGEMA:
                teammates[i] = _PogemaAStarAgent()
            else:
                teammates[i] = GreedyAgent(env, idle_idx, idx2move)

        # 目标 & 初始状态
        try:
            goal = tuple(env.goals[target_idx])
        except Exception:
            try:
                goal = tuple(obs[target_idx]["global_goal"])
            except Exception:
                goal = tuple(obs[target_idx]["global_xy"])
        state = obs[target_idx]

        # 动态步数上限
        try:
            opt_len = max(1, len(env.global_guidance[target_idx]) + 1)
        except Exception:
            try:
                sx, sy = state['global_xy']; gx, gy = goal
                opt_len = max(1, abs(sx - gx) + abs(sy - gy) + 1)
            except Exception:
                opt_len = 60

        cfg_max = int(cfg.get("max_episode_steps", 10**9))
        env_max = int(getattr(env, "max_episode_steps", cfg_max))
        mult = 8 if stage_name in ("0","1") else (10 if stage_name in ("2a","2b") else 12)
        timesteps_per_episode = min(env_max, max(80, int(opt_len * mult)))

        # cpx≈0.102 放宽时间
        if cpx_val is not None and np.isfinite(cpx_val) and abs(float(cpx_val) - 0.102) <= 0.008:
            timesteps_per_episode = int(timesteps_per_episode * 1.2)
            used_max_seconds = max(used_max_seconds, 75)

        model.eval()
        episode_traj = []
        episode_start_time = time.time()
        success_flag = False
        no_move_steps = 0
        last_pos_for_stuck = tuple(state.get('global_xy', (None, None))) if isinstance(state, dict) else None
        prev_action = None

        # —— 奖励塑形系数（子阶段强 → 弱）
        if stage_name in ("2a","3a","4a"):
            K = {"pre_col": -0.20, "wait": 0.06, "osc": -0.06, "time": -0.0005, "crowd": -0.04}
        elif stage_name in ("2b","3b","4b"):
            K = {"pre_col": -0.14, "wait": 0.04, "osc": -0.04, "time": -0.0007, "crowd": -0.03}
        else:
            K = {"pre_col": -0.12, "wait": 0.03, "osc": -0.03, "time": -0.0010, "crowd": -0.02}

        # 局部拥堵计数（半径2）
        def _local_congestion(obs_all, idx, radius=2):
            try:
                x, y = obs_all[idx]['global_xy']
            except Exception:
                return 0
            cnt = 0
            for j, o in enumerate(obs_all):
                if j == idx: continue
                try:
                    xx, yy = o['global_xy']
                    if abs(xx - x) + abs(yy - y) <= radius:
                        cnt += 1
                except Exception:
                    pass
            return cnt

        for t in range(timesteps_per_episode):
            if time.time() - episode_start_time > used_max_seconds:
                _pwrite(pbar, f"⏰ Episode {episode} 超时（>{used_max_seconds}s），终止本集")
                break

            prev_pos = tuple(state.get('global_xy', (0, 0))) if isinstance(state, dict) else None

            # —— 动作选择（含风险感知）
            actions = [idle_idx] * n_agents_pg
            for i in range(n_agents_pg):
                if i == target_idx:
                    x = _obs_to_tensor_CDHW(obs[i], device, expected_c=in_channels)  # [1,C,D,H,W]
                    eps = float(getattr(agent, "epsilon", 0.1))
                    if random.random() < eps:
                        a = random.randrange(n_actions)
                    else:
                        with torch.no_grad():
                            q = agent.q_network(x)                 # [1, A]
                            # 无效动作屏蔽
                            try:
                                invalid, idx2move = _invalid_actions_for(env, obs[i], idle_idx)
                                if invalid:
                                    q[0, invalid] = -1e9
                            except Exception:
                                idx2move = {}
                            # 风险感知：预估每个动作的冲突风险，q - w*risk
                            if risk_weight > 1e-8 and idx2move:
                                scores = q.clone()
                                base = [idle_idx] * n_agents_pg
                                for a_idx in range(n_actions):
                                    base[i] = a_idx
                                    try:
                                        risk = 1.0 if will_conflict_next_step(obs, base, i, idx2move) else 0.0
                                    except Exception:
                                        risk = 0.0
                                    scores[0, a_idx] = q[0, a_idx] - min(risk_clip, max(0.0, risk)) * risk_weight
                                a = int(torch.argmax(scores, dim=1))
                            else:
                                a = int(torch.argmax(q, dim=1))
                    actions[i] = a
                else:
                    try:
                        if HAVE_POGEMA:
                            a_tm = int(teammates[i].act(obs[i]))
                        else:
                            a_tm = int(teammates[i].act(obs[i], env.goals[i]))
                    except Exception:
                        a_tm = idle_idx
                    # 10% 怠速扰动（让路/降拥堵）
                    if random.random() < 0.10:
                        a_tm = idle_idx
                    actions[i] = a_tm

            # —— 预判奖励塑形（去冲突/少回头/拥堵等待）
            osc_pen = 0.0
            pre_collision_pen = 0.0
            wait_bonus = 0.0
            crowd_pen = 0.0

            try:
                if prev_action is not None:
                    opp = opposite_of(prev_action, idx2move, idle_idx)
                    if actions[target_idx] in opp:
                        osc_pen = K["osc"]
            except Exception:
                pass

            try:
                if will_conflict_next_step(obs, actions, target_idx, idx2move):
                    pre_collision_pen = K["pre_col"]
            except Exception:
                pass

            # 拥堵/等待塑形：如果局部拥堵且当前选择 idle → 小额奖励；局部很挤且继续推进 → 轻微惩罚
            try:
                cong = _local_congestion(obs, target_idx, radius=2)
                chose_idle = (actions[target_idx] == idle_idx)
                if cong >= 3 and chose_idle:
                    wait_bonus = K["wait"]
                elif cong >= 4 and not chose_idle:
                    crowd_pen = K["crowd"]
            except Exception:
                pass

            prev_action = actions[target_idx]

            # —— 真正执行一步
            obs, reward, terminated, truncated, info = env.step(tuple(int(a) for a in actions))

            # 注入塑形
            try:
                reward[target_idx] += (osc_pen + pre_collision_pen + wait_bonus + crowd_pen)
            except Exception:
                pass

            # 稠密/时间/碰撞
            try:
                ax, ay = obs[target_idx]['global_xy']; gx, gy = goal
                if prev_pos is not None:
                    px, py = prev_pos
                    d_prev = abs(px - gx) + abs(py - gy)
                    d_now  = abs(ax - gx) + abs(ay - gy)
                    r_dense = 0.05 * (d_prev - d_now)
                else:
                    r_dense = 0.0

                collided = False
                try:
                    if isinstance(info, dict):
                        col = info.get("collision", None)
                        if isinstance(col, (list, tuple)):
                            collided = bool(col[target_idx])
                        elif isinstance(col, dict):
                            collided = bool(col.get(target_idx, False))
                except Exception:
                    pass
                r_coll = -0.2 if collided else 0.0
                r_time = K["time"]

                reward[target_idx] += (r_dense + r_coll + r_time)
            except Exception:
                pass

            # 到达/终止
            try:
                agent_pos = tuple(obs[target_idx]['global_xy'])
            except Exception:
                agent_pos = tuple(state.get('global_xy', (0, 0)))
            done = (agent_pos == goal)
            terminated[target_idx] = done

            # STUCK 终止
            if STUCK_PATIENCE > 0:
                if last_pos_for_stuck == agent_pos:
                    no_move_steps += 1
                    if no_move_steps >= STUCK_PATIENCE:
                        break
                else:
                    no_move_steps = 0
                    last_pos_for_stuck = agent_pos

            trans = (
                state,
                actions[target_idx],
                reward[target_idx],
                obs[target_idx],
                terminated[target_idx],
            )
            agent.store(*trans)
            state = obs[target_idx]
            episode_traj.append(trans)

            if done:
                success_flag = True
                break

            # —— 回放训练（支持 PER / 非 PER）
            try:
                rb = getattr(agent, "replay_buffer", None)
                if rb is None:
                    pass
                elif _rb_is_iterable(rb):
                    rb_list = list(rb)
                    n = len(rb_list)
                    if n >= used_batch_size:
                        recent_cap = 12000
                        split = max(0, n - recent_cap)
                        older, recent = rb_list[:split], rb_list[split:]

                        # 用阶段名覆盖的 mix
                        if replay_mix:
                            ps, pr, ph = replay_mix
                        else:
                            # 默认
                            ps, pr, ph = (0.30, 0.70, 0.00)

                        n_succ   = int(used_batch_size * ps)
                        n_recent = int(used_batch_size * pr)
                        n_hard   = max(0, used_batch_size - n_succ - n_recent)

                        batch = []
                        if recent:
                            batch += random.sample(recent, min(len(recent), n_recent))
                        if len(batch) < n_recent and older:
                            need = n_recent - len(batch)
                            batch += random.sample(older, min(len(older), need))

                        if hasattr(agent, "success_buffer") and agent.success_buffer and n_succ > 0:
                            succ_pool = list(agent.success_buffer)
                            if succ_pool:
                                batch += random.sample(succ_pool, min(len(succ_pool), n_succ))

                        def _dist_to_goal_state(st):
                            try:
                                ax, ay = st['global_xy']; gx, gy = goal
                                return abs(ax-gx)+abs(ay-gy)
                            except Exception:
                                return 1e9
                        if n_hard > 0:
                            pool_all = (recent or []) + (older or [])
                            pool_all.sort(key=lambda tr: _dist_to_goal_state(tr[0]))
                            hard = []
                            for tr in pool_all:
                                if not bool(tr[4]):  # 未完成
                                    hard.append(tr)
                                if len(hard) >= n_hard:
                                    break
                            batch += hard

                        random.shuffle(batch)
                        batch = batch[:used_batch_size]

                        if len(batch) >= used_batch_size:
                            if hasattr(agent, "retrain_from_transitions") and callable(agent.retrain_from_transitions):
                                _ = agent.retrain_from_transitions(batch)
                            else:
                                _ = agent.retrain(used_batch_size)
                else:
                    if hasattr(rb, "__len__") and len(rb) >= used_batch_size:
                        _ = agent.retrain(used_batch_size)
            except Exception:
                try:
                    if hasattr(agent, "replay_buffer") and len(agent.replay_buffer) >= used_batch_size:
                        _ = agent.retrain(used_batch_size)
                except Exception:
                    pass

        # 成功缓存
        if success_flag and hasattr(agent, "success_buffer"):
            for t_ in episode_traj:
                agent.success_buffer.append(t_)

        # 4) 统计 & 日志
        if success_flag:
            success_count_total += 1
        episode += 1

        scheduler.add_episode_result(int(success_flag), pbar)

        success_rate = success_count_total / max(1, episode)
        writer.add_scalar('SuccessRate/global', success_rate, episode)
        writer.add_scalar('success', 1 if success_flag else 0, episode)
        writer.add_scalar('SuccessRate/stage', scheduler.stage_sr(), episode)

        if pbar is not None:
            pbar.set_postfix(
                Stage=stage_name,
                success=int(success_flag),
                sr_global=f"{success_rate:.3f}",
                sr_stage=f"{scheduler.stage_sr():.3f}"
            )
            pbar.update(1)

        print(f"[Episode {episode}] Stage {stage_name} | Success={'✅' if success_flag else '❌'} | "
              f"SR_global={success_rate:.3f}, SR_stage={scheduler.stage_sr():.6f}")

        training_logs.append({
            "episode": episode,
            "stage": stage_name,
            "map": map_name,
            "agents": _pogema_num_agents_from_obs(obs),
            "complexity": (float(cpx_val) if cpx_val is not None else np.nan),
            "success": int(success_flag),
            "success_rate_global": float(success_rate),
            "success_rate_stage": float(scheduler.stage_sr()),
        })

        # 晋级/重复判定
        if scheduler.should_advance():
            # 保存阶段权重
            try:
                ckpt_path = Path("models") / f"stage_{stage_name}_model.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path.as_posix())
                _pwrite(pbar, f"💾 阶段权重已保存：{ckpt_path}")
            except Exception as e:
                _pwrite(pbar, f"⚠️ 保存阶段权重失败：{e}")

            scheduler.advance(pbar)
            # 进入新阶段：epsilon 回弹
            if hasattr(agent, "epsilon"):
                floors = {"2a":0.35,"2b":0.28,"3a":0.28,"3b":0.22,"4a":0.20,"4b":0.16}
                agent.epsilon = max(floors.get(stage_name, 0.10), float(getattr(agent, "epsilon", 0.1)))
            continue
        else:
            if scheduler._ep_in_stage >= scheduler.min_episodes_per_stage:
                scheduler.repeat_stage(pbar)

    # —— 收尾
    df = pd.DataFrame(training_logs)
    out_dir = Path(run_dir) / "episodes"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "episodes_refined.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"📝 每集日志已保存：{csv_path}")
    _pwrite(pbar, "🎉 所有阶段训练完成！")

    writer.close()
    return agent


# ================== 主程序 ==================
if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 1) 读 YAML
    with open(MAP_SETTINGS_PATH, "r", encoding="utf-8") as f:
        base_map_settings = yaml.safe_load(f)
    if isinstance(base_map_settings, list):
        base_map_settings = {(m.get("name") or f"map_{i}"): m for i, m in enumerate(base_map_settings)}

    # 2) 合并复杂度
    base_map_settings = _merge_complexity_from_csv(base_map_settings)

    # 3) 构建 Scheduler（细化后共 8 阶）
    scheduler = ComplexityScheduler(
        base_map_settings=base_map_settings,
        min_per_stage=MIN_PER_STAGE,
        min_episodes_per_stage=MIN_EPISODES_PER_STAGE,
        threshold=THRESHOLD,
        window_size=WINDOW_SIZE,
        shuffle_each_stage=True,
        seed=0,
    )

    # 4) 设备 & 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNNModel().to(device)

    # 5) 训练（注意：map_settings 只给一个占位，后续每集 scheduler 会切换）
    init_map = scheduler.get_updated_map_settings()
    agent = train(
        model=model,
        scheduler=scheduler,
        map_settings=init_map,
        map_probs=None,
        batch_size=BATCH_SIZE,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        decay_range=DECAY_RANGE,
        log_dir=LOG_DIR,
        device=device,
        max_episode_seconds=MAX_EPISODE_SECONDS,
        run_dir=RUN_DIR,
    )

    # 6) 保存最终模型
    out_path = Path(MODEL_OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path.as_posix())
    print(f"✅ 模型已保存到 {out_path}")
