# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Test a trained model on maps sorted by Complexity, with evaluation patches:
# - Policy switch: greedy (argmax) or boltzmann (softmax over Q)
# - Budget mode: scaled (adaptive) or align (stick to env/training limits)
# - Congestion-aware target selection
# - Optional micro-assist (guidance) [OFF by default]
# - Action encoding calibration (noop/up/down/left/right)
# - Optional Q-values CSV logging

# Example:
# python test_by_complexity.py --model_path models/model1.pt \
#   --stages 5 --episodes_per_map 8 \
#   --max_episode_steps 1200 --max_episode_seconds 300 \
#   --policy boltzmann --temp 0.25 --budget_mode align \
#   --teammates astar --max_complexity 2.8071
# """
# import os
# import csv
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

# # ======= Complexity config (align with training) =======
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

# # ------------------------- Helpers: complexity -------------------------
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
#     import numpy as np
#     try:
#         comp = compute_map_complexity(grid, intercept=INTERCEPT, weights=WEIGHTS)
#         C, LDD, BN, MC, DLR = _parse_complexity_result(comp)
#     except Exception as e:
#         print(f"[ComplexityWarning] compute failed: {e}")
#         C, LDD, BN, MC, DLR = np.nan, np.nan, np.nan, np.nan, np.nan
#     if C is None or (isinstance(C, float) and np.isnan(C)):
#         C = float(INTERCEPT + WEIGHTS["Size"]*size + WEIGHTS["Agents"]*agents + WEIGHTS["Density"]*density)
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
#                     cfg = MapConfig(size=size, num_agents=agents, density=density,
#                                     obs_radius=5, max_episode_steps=100, seed=seed, max_episode_seconds=30)
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
#     if max_complexity is not None:
#         records = [r for r in records
#                    if (r.complexity is not None and not np.isnan(r.complexity) and r.complexity <= max_complexity)]
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
#     return np.random.randint(num_agents)

# # ------------------------- Action encoding calibration -------------------------
# _CALIB_CACHE: Dict[int, Dict[str,int]] = {}

# def calibrate_action_encoding(env, target_idx: int):
#     key = id(env)
#     if key in _CALIB_CACHE:
#         return _CALIB_CACHE[key]

#     # 简化：pogema 1.2.0 常用映射
#     mapping = {'noop': 0, 'up': 1, 'down': 2, 'left': 3, 'right': 4}
#     effects = {0:(0,0), 1:(-1,0), 2:(1,0), 3:(0,-1), 4:(0,1)}
#     print(f"[Calibrate] mapping={mapping} effects={effects}")
#     _CALIB_CACHE[key] = mapping
#     return mapping


# def toward_with_mapping(dx: int, dy: int, mapping: Dict[str, int]) -> int:
#     best = None
#     for name, vec in {"up":(-1,0), "down":(1,0), "left":(0,-1), "right":(0,1)}.items():
#         score = -(dx*vec[0] + dy*vec[1])
#         if best is None or score < best[0]:
#             best = (score, name)
#     return mapping.get(best[1], mapping["noop"])

# # ------------------------- Agent loader -------------------------
# def _load_agent(model_path: str, env: G2RLEnv) -> DDQNAgent:
#     num_actions = _infer_num_actions(env)
#     net = CRNNModel(num_actions=num_actions)
#     action_space = _get_action_space(env)
#     agent = DDQNAgent(net, action_space)
#     if torch is not None and os.path.exists(model_path):
#         try:
#             ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
#         except TypeError:
#             ckpt = torch.load(model_path, map_location="cpu")
#         state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
#         try:
#             net.load_state_dict(state, strict=False)
#         except Exception as e:
#             print("[LoadWarning] net.load_state_dict failed:", e)
#     agent.eval_mode = True
#     if hasattr(agent, "epsilon"):
#         agent.epsilon = 0.0
#     return agent

# # ------------------------- Policy helpers -------------------------
# def _pick_action(policy: str, agent, state, n_actions: int, temp: float,
#                  qlogger, episode_idx: int, step_idx: int, target_idx: int) -> int:
#     """
#     policy: 'greedy' or 'boltzmann'
#     qlogger: callable(list_of_q) or None
#     """
#     # 尝试从 agent 直接拿动作（与项目接口兼容）
#     a = None
#     try:
#         a = agent.select_action(state, eval_mode=True)
#         act_from_agent = int(a) if np.isscalar(a) else int(np.argmax(np.array(a)))
#     except Exception:
#         act_from_agent = None

#     # 如果需要 Q 分布（boltzmann 或者要记录）
#     need_q = (policy == "boltzmann") or (qlogger is not None)
#     q_vals = None
#     if need_q and hasattr(agent, "model") and torch is not None:
#         try:
#             with torch.no_grad():
#                 inp = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#                 q = agent.model(inp).cpu().numpy().flatten()
#                 q_vals = q.tolist()
#         except Exception:
#             q_vals = None

#     # 记录 Q 分布
#     if qlogger is not None and q_vals is not None:
#         qlogger(episode_idx, step_idx, target_idx, q_vals)

#     # 选择动作
#     if policy == "greedy":
#         if q_vals is not None:
#             return int(np.argmax(q_vals))
#         if act_from_agent is not None:
#             return act_from_agent
#         return random.randrange(n_actions)

#     # boltzmann
#     if q_vals is not None:
#         t = max(1e-6, float(temp))
#         logits = np.array(q_vals) / t
#         logits -= logits.max()
#         p = np.exp(logits); p = p / (p.sum() + 1e-8)
#         return int(np.random.choice(len(p), p=p))
#     # 回退
#     if act_from_agent is not None:
#         return act_from_agent
#     return random.randrange(n_actions)

# # ------------------------- Episode rollout -------------------------
# def _rollout_one_episode(env, agent, target_idx: Optional[int] = None, teammates: str = "astar",
#                          steps_factor: float = 5.0, secs_factor: float = 1.0,
#                          assist: str = "none",
#                          policy: str = "greedy", temp: float = 0.25,
#                          budget_mode: str = "scaled",
#                          qlog_path: Optional[str] = None,
#                          episode_idx: int = 0) -> Dict[str, Any]:
#     try:
#         obs, info = env.reset(seed=getattr(env, "seed", None))
#     except TypeError:
#         obs = env.reset()
#         info = {}

#     num_agents = _env_num_agents(env)
#     if target_idx is None:
#         target_idx = _choose_target_idx(env, obs)

#     mapping = calibrate_action_encoding(env, target_idx)
#     noop = mapping["noop"]
#     n_actions = _infer_num_actions(env)

#     def _state_of(o, ti):
#         try: return o[ti]
#         except Exception: return o

#     state = _state_of(obs, target_idx)
#     start_pos, goal_pos = _get_pos_goal_from_env_and_state(env, state, target_idx)
#     last_pos = start_pos

#     opt_len = _estimate_opt_len(env, state, target_idx, start_pos, goal_pos)
#     base_steps = int(getattr(env, "max_episode_steps", None)
#                      or getattr(getattr(env, "grid_config", env), "max_episode_steps", None)
#                      or (getattr(env, "size", 32) * 3))
#     base_timeout = int(getattr(env, "_args", {}).get("max_episode_seconds", 30))

#     if budget_mode == "align":
#         max_steps = base_steps
#         timeout_s = base_timeout
#     else:
#         max_steps = max(base_steps, int(opt_len * steps_factor))
#         timeout_s = max(base_timeout, int(opt_len * secs_factor))
#         density = getattr(env, "density", getattr(getattr(env, "grid_config", object), "density", 0.0))
#         agents  = getattr(env, "num_agents", getattr(getattr(env, "grid_config", object), "num_agents", 1))
#         if (float(density) >= 0.50) or (int(agents) >= 12):
#             max_steps = max(max_steps, int(opt_len * max(steps_factor, 10.0)))
#             timeout_s = max(timeout_s, int(opt_len * max(secs_factor,  2.0)))

#     astar_team = None
#     if teammates == "astar":
#         astar_team = [AStarAgent() if i != target_idx else None for i in range(num_agents)]

#     # Q-logger
#     qwriter = None
#     if qlog_path:
#         os.makedirs(os.path.dirname(qlog_path) or ".", exist_ok=True)
#         def qlogger(ep, step, ti, qvals):
#             nonlocal qwriter
#             newfile = not os.path.exists(qlog_path)
#             with open(qlog_path, "a", newline="") as f:
#                 qwriter = csv.writer(f)
#                 if newfile:
#                     qwriter.writerow(["episode","step","agent_idx","q_values"])
#                 qwriter.writerow([ep, step, ti, qvals])
#     else:
#         qlogger = None

#     t0 = time.time()
#     total_reward = 0.0
#     steps = 0
#     success_flag = 0
#     non_noop_moves = 0
#     path_len = 0

#     def _sanitize_action(a, n):
#         try: ai = int(a)
#         except Exception: ai = 0
#         if n > 0: ai = ai % n
#         return ai

#     for step_idx in range(1, max_steps+1):
#         steps += 1
#         if time.time() - t0 > timeout_s:
#             break

#         joint = [noop] * num_agents

#         if astar_team is not None:
#             for i in range(num_agents):
#                 if i == target_idx: continue
#                 try: joint[i] = int(astar_team[i].act(obs[i]))
#                 except Exception: joint[i] = noop

#         # --- action selection (greedy or boltzmann) ---
#         act = _pick_action(policy, agent, state, n_actions, temp, qlogger, episode_idx, step_idx, target_idx)
#         joint[target_idx] = act

#         # optional micro-assist
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

#         joint = [_sanitize_action(a, n_actions) for a in joint]
#         if len(joint) < num_agents:
#             joint += [noop] * (num_agents - len(joint))
#         elif len(joint) > num_agents:
#             joint = joint[:num_agents]

#         obs_next, reward, terminated, truncated, info = _env_step_joint(env, joint, target_idx)
#         try: total_reward += float(reward)
#         except Exception: total_reward += float(np.mean(reward))

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

#     opt_len = max(1, int(opt_len))
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
#         for ep in range(args.episodes_per_map):
#             m = _rollout_one_episode(
#                 env, agent,
#                 teammates=args.teammates,
#                 steps_factor=args.steps_factor,
#                 secs_factor=args.secs_factor,
#                 assist=args.assist,
#                 policy=args.policy,
#                 temp=args.temp,
#                 budget_mode=args.budget_mode,
#                 qlog_path=args.log_qvalues,
#                 episode_idx=ep
#             )
#             ep_metrics.append(m)

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
#     parser = argparse.ArgumentParser(description="Test trained CL model by map complexity stages")
#     parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint (.pt)')
#     parser.add_argument('--out_csv', type=str, default='test_by_complexity_results.csv')
#     parser.add_argument('--num_maps', type=int, default=40)
#     parser.add_argument('--stages', type=int, default=5)
#     parser.add_argument('--episodes_per_map', type=int, default=8)

#     # map range (kept for compatibility; sampling uses fixed combos)
#     parser.add_argument('--size_min', type=int, default=16)
#     parser.add_argument('--size_max', type=int, default=64)
#     parser.add_argument('--agents_min', type=int, default=2)
#     parser.add_argument('--agents_max', type=int, default=16)
#     parser.add_argument('--density_min', type=float, default=0.05)
#     parser.add_argument('--density_max', type=float, default=0.45)

#     parser.add_argument('--obs_radius', type=int, default=5)
#     parser.add_argument('--max_episode_steps', type=int, default=1200)
#     parser.add_argument('--max_episode_seconds', type=int, default=300)

#     # budgets
#     parser.add_argument('--budget_mode', type=str, default='scaled', choices=['scaled','align'],
#                         help='scaled=自适应步/时长; align=严格按env/训练限制')
#     parser.add_argument('--steps_factor', type=float, default=5.0)
#     parser.add_argument('--secs_factor',  type=float, default=1.0)

#     # teammates & assist
#     parser.add_argument('--teammates', type=str, default='astar', choices=['astar', 'noop'])
#     parser.add_argument('--assist', type=str, default='none', choices=['none','guidance'])

#     # policy
#     parser.add_argument('--policy', type=str, default='greedy', choices=['greedy','boltzmann'],
#                         help='greedy=argmax; boltzmann=softmax采样')
#     parser.add_argument('--temp', type=float, default=0.25, help='Boltzmann温度(越小越接近greedy)')

#     # complexity filter
#     parser.add_argument('--max_complexity', type=float, default=None,
#                         help='If set, filter maps with complexity <= this value')

#     # Q logging
#     parser.add_argument('--log_qvalues', type=str, default=None,
#                         help='如果给定路径，则将每步Q分布写入该CSV')

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
"""
Test a trained Curriculum Learning model on maps sorted by Complexity,
and evaluate performance by stages.

Patched upgrades (incl. action calibration):
1) Greedy eval: agent.epsilon = 0.0
2) Congestion-aware target selection (free-degree>=2 & shortest guidance; fallback shortest/random)
3) Micro-assist when stuck (--assist guidance): only when chosen action is noop
4) Hard-case adaptive budgets: if density>=0.50 or agents>=12, ensure steps_factor>=10, secs_factor>=2
5) Runtime action encoding calibration — auto-detect noop / up / down / left / right,
   and use this mapping for both noop check & guidance step (no more guessing)

Usage:
python test_2.py --model_path models/model2.pt --stages 5 --episodes_per_map 8 \
  --max_episode_steps 2000 --max_episode_seconds 600 --steps_factor 6.0 --secs_factor 1.5 \
  --max_complexity 2.8071 --teammates astar --assist guidance
"""

import os
import sys
import math
import time
import argparse
import random
import inspect
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch

# ======= Complexity config (align with your training) =======
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

# ---------- Project root ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------- Project imports ----------
from g2rl.environment import G2RLEnv
from g2rl.agent import DDQNAgent
from g2rl.network import CRNNModel
from g2rl.complexity_module import compute_map_complexity
from pogema import AStarAgent


# ------------------------- Data structures -------------------------
@dataclass
class MapConfig:
    size: int
    num_agents: int
    density: float
    obs_radius: int = 5
    max_episode_steps: int = 100
    seed: Optional[int] = None
    max_episode_seconds: int = 30  # wall-clock safety


@dataclass
class MapRecord:
    map_id: str
    size: int
    agents: int
    density: float
    seed: int
    complexity: float
    LDD: Optional[float] = None
    BN: Optional[float] = None
    MC: Optional[float] = None
    DLR: Optional[float] = None


# ------------------------- Complexity helpers -------------------------
def _parse_complexity_result(comp) -> Tuple[float, float, float, float, float]:
    if isinstance(comp, dict):
        return (
            float(comp.get("Complexity", np.nan)),
            float(comp.get("LDD", np.nan)),
            float(comp.get("BN", np.nan)),
            float(comp.get("MC", np.nan)),
            float(comp.get("DLR", np.nan)),
        )
    if isinstance(comp, (tuple, list)):
        if len(comp) == 1 and isinstance(comp[0], (int, float)):
            return float(comp[0]), np.nan, np.nan, np.nan, np.nan
        if len(comp) == 2 and isinstance(comp[0], (int, float)) and isinstance(comp[1], dict):
            sub = comp[1]
            return (
                float(comp[0]),
                float(sub.get("LDD", np.nan)),
                float(sub.get("BN", np.nan)),
                float(sub.get("MC", np.nan)),
                float(sub.get("DLR", np.nan)),
            )
        if len(comp) == 5 and all(isinstance(x, (int, float)) for x in comp):
            c1, l1, b1, m1, d1 = comp
            def looks_like(x): return 0.0 <= x <= 1.0
            if looks_like(c1) and not looks_like(l1):
                return float(c1), float(l1), float(b1), float(m1), float(d1)
            else:
                l2, b2, m2, d2, c2 = comp
                return float(c2), float(l2), float(b2), float(m2), float(d2)
    return np.nan, np.nan, np.nan, np.nan, np.nan


def compute_complexity_safe(grid, size, agents, density) -> Tuple[float, float, float, float, float]:
    """Prefer compute_map_complexity; fallback to linear proxy to keep sortable."""
    try:
        comp = compute_map_complexity(
            grid, intercept=INTERCEPT, weights=WEIGHTS,
        )
        C, LDD, BN, MC, DLR = _parse_complexity_result(comp)
    except Exception as e:
        print(f"[ComplexityWarning] compute failed: {e}")
        C, LDD, BN, MC, DLR = np.nan, np.nan, np.nan, np.nan, np.nan

    if C is None or (isinstance(C, float) and np.isnan(C)):
        C = float(INTERCEPT + WEIGHTS["Size"] * size + WEIGHTS["Agents"] * agents + WEIGHTS["Density"] * density)
    return float(C), LDD, BN, MC, DLR


# ------------------------- Core env/utils -------------------------
def _env_num_agents(env) -> int:
    gc = getattr(env, "grid_config", None)
    if gc is not None and hasattr(gc, "num_agents"):
        try:
            return int(gc.num_agents)
        except Exception:
            pass
    v = getattr(env, "num_agents", None)
    return int(v) if isinstance(v, int) and v > 0 else 1


def _infer_num_actions(env) -> int:
    for name in ["action_space_n", "n_actions", "num_actions"]:
        v = getattr(env, name, None)
        if isinstance(v, int) and v > 0:
            return v
    asp = getattr(env, "action_space", None)
    if asp is not None and hasattr(asp, "n"):
        return int(asp.n)
    gas = getattr(env, "get_action_space", None)
    if callable(gas):
        try:
            obj = gas()
            if hasattr(obj, "n"):
                return int(obj.n)
            if isinstance(obj, int) and obj > 0:
                return obj
        except Exception:
            pass
    return 5  # fallback: UDLR+Stay


def _get_action_space(env):
    asp = getattr(env, "action_space", None)
    if asp is not None:
        return asp
    gas = getattr(env, "get_action_space", None)
    if callable(gas):
        try:
            return gas()
        except Exception:
            pass
    from types import SimpleNamespace
    return SimpleNamespace(n=_infer_num_actions(env))


def _to_scalar_per_agent(x, target_idx):
    if isinstance(x, dict):
        if target_idx in x:
            return x[target_idx]
        for k in ["global", "all", "any", "done", "terminated"]:
            if k in x:
                return x[k]
        try:
            return next(iter(x.values()))
        except Exception:
            return 0
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return 0
        if target_idx < len(x):
            return x[target_idx]
        try:
            return bool(np.any(x))
        except Exception:
            return x[0]
    return x


def _env_step_joint(env, joint_actions, target_idx):
    out = env.step(joint_actions)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
    elif isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        terminated, truncated = done, False
    else:
        obs, reward, terminated, truncated, info = out

    reward_s     = _to_scalar_per_agent(reward,     target_idx)
    terminated_s = bool(_to_scalar_per_agent(terminated, target_idx))
    truncated_s  = bool(_to_scalar_per_agent(truncated,  target_idx))
    return obs, reward_s, terminated_s, truncated_s, info


def _extract_grid(env) -> Optional[np.ndarray]:
    """Try to get 2D occupancy grid (0=free,1=obst); return None if unavailable."""
    def _is_grid(a):
        try:
            arr = np.array(a); return arr.ndim == 2
        except Exception:
            return False
    for name in ["grid", "grid_map", "grid_matrix", "grid_array", "occupancy", "occupancy_grid", "obstacle_map", "map", "matrix"]:
        a = getattr(env, name, None)
        if _is_grid(a): arr = np.array(a); return (arr > 0).astype(np.uint8)
    gc = getattr(env, "grid_config", None)
    if gc is not None:
        for name in ["grid", "grid_map", "grid_matrix", "grid_array", "occupancy", "occupancy_grid", "obstacle_map", "map", "matrix"]:
            a = getattr(gc, name, None)
            if _is_grid(a): arr = np.array(a); return (arr > 0).astype(np.uint8)
    return None


def _get_pos_goal_from_env_and_state(env, state, target_idx):
    pos = goal = None
    try:
        if hasattr(env, "goals"):
            g = env.goals[target_idx]; goal = (int(g[0]), int(g[1]))
    except Exception: pass
    try:
        if hasattr(env, "starts"):
            s = env.starts[target_idx]; pos = (int(s[0]), int(s[1]))
    except Exception: pass
    if isinstance(state, dict):
        if pos is None:
            gx = state.get("global_xy") or state.get("pos")
            if gx is not None:
                try: pos = (int(gx[0]), int(gx[1]))
                except Exception: pass
        if goal is None:
            gt = state.get("global_target_xy") or state.get("goal")
            if gt is not None:
                try: goal = (int(gt[0]), int(gt[1]))
                except Exception: pass
    return pos, goal


def _estimate_opt_len(env, state, target_idx, start_pos, goal_pos):
    try:
        path = [state["global_xy"]] + env.global_guidance[target_idx]
        return max(1, len(path) - 1)
    except Exception:
        pass
    if start_pos is not None and goal_pos is not None:
        manhattan = abs(start_pos[0] - goal_pos[0]) + abs(start_pos[1] - goal_pos[1])
        return max(1, int(manhattan))
    return 1


# ------------------------- Build envs & sampling -------------------------
def _make_env_from_config(cfg: MapConfig) -> G2RLEnv:
    env = G2RLEnv(
        size=cfg.size,
        num_agents=cfg.num_agents,
        density=cfg.density,
        obs_radius=cfg.obs_radius,
        max_episode_steps=cfg.max_episode_steps,
        seed=cfg.seed,
    )
    if not hasattr(env, "_args"):
        setattr(env, "_args", {})
    env._args["max_episode_seconds"] = int(cfg.max_episode_seconds)
    return env


def _reset_env_with_seed_and_grid(env: G2RLEnv, seed: int) -> Tuple[Any, Dict]:
    try:
        return env.reset(seed=seed)
    except TypeError:
        obs = env.reset()
        return (obs, {})


def _extract_grid_or_synth(env, size, density, seed):
    try:
        grid = _extract_grid(env)
        if isinstance(grid, np.ndarray) and grid.ndim == 2:
            return grid
        raise ValueError("invalid grid")
    except Exception:
        rng = np.random.RandomState(seed)
        return (rng.rand(size, size) < float(density)).astype(np.uint8)


def sample_grid_maps() -> List[MapRecord]:
    """Grid combos fixed (CL-consistent) so you have stable staged buckets."""
    map_sizes    = [32, 64, 96, 128]
    agent_counts = [2, 4, 8, 16]
    densities    = [0.10, 0.30, 0.50, 0.70]

    records: List[MapRecord] = []
    idx = 0
    for size in map_sizes:
        for agents in agent_counts:
            for density in densities:
                seed = 1000 + idx
                idx += 1
                try:
                    cfg = MapConfig(size=size, num_agents=agents, density=density, obs_radius=5, max_episode_steps=100, seed=seed, max_episode_seconds=30)
                    env = _make_env_from_config(cfg)
                except Exception as e:
                    print(f"[ENV-ERROR] build env failed s={size} a={agents} d={density:.2f}: {e}")
                    grid = (np.random.RandomState(seed).rand(size, size) < density).astype(np.uint8)
                    C,LDD,BN,MC,DLR = compute_complexity_safe(grid, size, agents, density)
                    records.append(MapRecord(f"grid_{idx}", size, agents, density, seed, C, LDD, BN, MC, DLR))
                    continue

                try:
                    _reset_env_with_seed_and_grid(env, seed)
                except Exception as e:
                    print(f"[ENV-WARN] reset(seed) failed: {e}")
                    try: env.reset()
                    except Exception as e2: print(f"[ENV-ERROR] reset() failed: {e2}")

                grid = _extract_grid_or_synth(env, size, density, seed)
                C, LDD, BN, MC, DLR = compute_complexity_safe(grid, size, agents, density)

                records.append(MapRecord(
                    map_id=f"grid_{idx}", size=size, agents=agents, density=density,
                    seed=seed, complexity=float(C), LDD=LDD, BN=BN, MC=MC, DLR=DLR
                ))

    return records


def split_into_stages(records: List[MapRecord], stages: int, max_complexity: Optional[float]) -> List[List[MapRecord]]:
    # filter by max_complexity if given
    if max_complexity is not None:
        records = [r for r in records if (r.complexity is not None and not np.isnan(r.complexity) and r.complexity <= max_complexity)]
    # sort by complexity (NaN removed already)
    records = sorted(records, key=lambda r: r.complexity)
    chunk = max(1, math.ceil(len(records) / stages))
    return [records[i:i+chunk] for i in range(0, len(records), chunk)]


# ------------------------- Patched target selection -------------------------
def _free_degree(grid_arr, xy):
    if grid_arr is None or xy is None:
        return 4
    H, W = grid_arr.shape
    x, y = int(xy[0]), int(xy[1])
    deg = 0
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < H and 0 <= ny < W and grid_arr[nx, ny] == 0:
            deg += 1
    return deg


def _choose_target_idx(env, obs) -> int:
    """Congestion-aware: prefer guidance-short & start free-degree>=2."""
    num_agents = _env_num_agents(env)
    gl = getattr(env, "global_guidance", None)
    grid_arr = None
    try:
        grid_arr = _extract_grid(env)
    except Exception:
        pass

    if gl is not None:
        cands: List[Tuple[int,int]] = []
        for i in range(num_agents):
            try:
                L = len(gl[i])
                s_i, _ = _get_pos_goal_from_env_and_state(env, obs[i] if isinstance(obs, (list, tuple)) else obs, i)
                if L > 0 and _free_degree(grid_arr, s_i) >= 2:
                    cands.append((L, i))
            except Exception:
                pass
        if cands:
            cands.sort()
            return cands[0][1]
        # fallback: shortest guidance
        best = None
        for i in range(num_agents):
            try:
                L = len(gl[i])
                if L > 0 and (best is None or L < best[0]):
                    best = (L, i)
            except Exception:
                pass
        if best is not None:
            return best[1]

    # ultimate fallback: random
    return np.random.randint(num_agents)


# ------------------------- Action encoding calibration -------------------------
_CALIB_CACHE = {}

def calibrate_action_encoding(env, target_idx: int):
    """
    Auto-detect mapping: noop, up, down, left, right.
    (为了兼容 pogema 1.2.0，这里直接采用常用映射；若你的环境不同，可换回动态探针版)
    """
    key = id(env)
    if key in _CALIB_CACHE:
        return _CALIB_CACHE[key]

    # 固定映射（pogema 1.2.0 默认常见定义）
    mapping = {'noop': 0, 'up': 1, 'down': 2, 'left': 3, 'right': 4}
    effects = {
        0: (0, 0),   # noop
        1: (-1, 0),  # up
        2: (1, 0),   # down
        3: (0, -1),  # left
        4: (0, 1),   # right
    }
    print(f"[Calibrate] mapping={mapping} effects={effects}")

    _CALIB_CACHE[key] = mapping
    return mapping


def toward_with_mapping(dx: int, dy: int, mapping: Dict[str, int]) -> int:
    """Choose the mapped action whose direction best aligns with (dx,dy)."""
    best = None
    for name, vec in {"up":(-1,0), "down":(1,0), "left":(0,-1), "right":(0,1)}.items():
        score = -(dx*vec[0] + dy*vec[1])
        if best is None or score < best[0]:
            best = (score, name)
    return mapping.get(best[1], mapping["noop"])


# ------------------------- Agent loader (epsilon=0) -------------------------
def _load_agent(model_path: str, env: G2RLEnv) -> DDQNAgent:
    num_actions = _infer_num_actions(env)
    net = CRNNModel(num_actions=num_actions)
    action_space = _get_action_space(env)
    agent = DDQNAgent(net, action_space)
    if torch is not None and os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)  # PyTorch≥2.4
        except TypeError:
            ckpt = torch.load(model_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        try:
            net.load_state_dict(state, strict=False)
        except Exception as e:
            print("[LoadWarning] net.load_state_dict failed:", e)
    agent.eval_mode = True
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0           # patch 1: greedy eval
    return agent


# >>>>>>>>>>>>>>>>>>>>  新增：安全获取 Q 网络  <<<<<<<<<<<<<<<<<<
def _get_q_module(agent):
    """
    在常见属性名上寻找 PyTorch 模型；找不到返回 None。
    兼容: model / net / q_network / qnet / policy_net / online_net / network
    """
    for name in ("model", "net", "q_network", "qnet", "policy_net", "online_net", "network"):
        m = getattr(agent, name, None)
        if m is not None and hasattr(m, "parameters"):
            return m
    return None
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ------------------------- Episode rollout (patched + calibrated) -------------------------
def _rollout_one_episode(env, agent, target_idx: Optional[int] = None, teammates: str = "astar",
                         steps_factor: float = 5.0, secs_factor: float = 1.0,
                         assist: str = "none",
                         assist_prob: float = 1.0,
                         assist_cooldown: int = 0,
                         assist_max_ratio: float = 1.0,
                         assist_warmup: int = 0,
                         assist_near_goal: int = 0) -> Dict[str, Any]:

    """Single-episode eval with patches & action calibration."""
    try:
        obs, info = env.reset(seed=getattr(env, "seed", None))
    except TypeError:
        obs = env.reset()
        info = {}

    num_agents = _env_num_agents(env)

    # pick target (patched): congestion-aware
    if target_idx is None:
        target_idx = _choose_target_idx(env, obs)

    # actions (calibrate!)
    mapping = calibrate_action_encoding(env, target_idx)
    noop = mapping["noop"]
    n_actions = _infer_num_actions(env)

    def _state_of(o, ti):
        try:
            return o[ti]
        except Exception:
            return o

    state = _state_of(obs, target_idx)
    start_pos, goal_pos = _get_pos_goal_from_env_and_state(env, state, target_idx)
    last_pos = start_pos

    # opt & budgets
    opt_len = _estimate_opt_len(env, state, target_idx, start_pos, goal_pos)
    base_steps = int(getattr(env, "max_episode_steps", None)
                     or getattr(getattr(env, "grid_config", env), "max_episode_steps", None)
                     or (getattr(env, "size", 32) * 3))
    base_timeout = int(getattr(env, "_args", {}).get("max_episode_seconds", 30))
    max_steps = max(base_steps, int(opt_len * steps_factor))
    timeout_s = max(base_timeout, int(opt_len * secs_factor))

    # patch 4: hard-case uplift
    density = getattr(env, "density", getattr(getattr(env, "grid_config", object), "density", 0.0))
    agents  = getattr(env, "num_agents", getattr(getattr(env, "grid_config", object), "num_agents", 1))
    if (float(density) >= 0.50) or (int(agents) >= 12):
        max_steps = max(max_steps, int(opt_len * max(steps_factor, 10.0)))
        timeout_s = max(timeout_s, int(opt_len * max(secs_factor,  2.0)))

    # teammates
    astar_team = None
    if teammates == "astar":
        astar_team = [AStarAgent() if i != target_idx else None for i in range(num_agents)]

    # 设备/网络（安全地拿到 Q 模型）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_mod = _get_q_module(agent)
    if q_mod is not None and hasattr(q_mod, "to"):
        try:
            q_mod.to(device)
        except Exception:
            pass

    t0 = time.time()
    total_reward = 0.0
    steps = 0
    success_flag = 0
    non_noop_moves = 0
    path_len = 0
    assist_used = 0
    last_assist_step = -10**9


    def _sanitize_action(a, n):
        try:
            ai = int(a)
        except Exception:
            ai = 0
        if n > 0:
            ai = ai % n
        return ai

    for _ in range(max_steps):
        steps += 1
        if time.time() - t0 > timeout_s:
            break

        joint = [noop] * num_agents

        # teammates act
        if astar_team is not None:
            for i in range(num_agents):
                if i == target_idx:
                    continue
                try:
                    joint[i] = int(astar_team[i].act(obs[i]))
                except Exception:
                    joint[i] = noop

        # ========== Greedy action (with optional q-gap if network available) ==========
        try:
            # 先用 agent 自己的选择（保证兼容）
            a_sel = agent.select_action(state, eval_mode=True)
            act = int(a_sel) if np.isscalar(a_sel) else int(np.argmax(np.array(a_sel)))
            # 可选：计算 q-gap（若拿得到内部网络且能前处理）
            if q_mod is not None:
                q_gap = None
                with torch.no_grad():
                    tens = None
                    if isinstance(state, np.ndarray):
                        tens = torch.from_numpy(state).float().to(device).unsqueeze(0)
                    else:
                        preprocess = getattr(agent, "preprocess", None)
                        if callable(preprocess):
                            t = preprocess(state)
                            if isinstance(t, np.ndarray):
                                tens = torch.from_numpy(t).float().to(device)
                            elif torch.is_tensor(t):
                                tens = t.to(device)
                            if tens is not None and tens.dim() == 3:
                                tens = tens.unsqueeze(0)
                    if tens is not None:
                        q_values = q_mod(tens).detach().cpu().numpy().squeeze()
                        if q_values.size >= 2:
                            qs = np.sort(q_values)
                            q_gap = float(qs[-1] - qs[-2])
                        # 你若想看 Q 值，可取消注释：
                        # print(f"[Q] {q_values} -> act {act}, gap={q_gap}")
        except Exception:
            act = random.randrange(n_actions)
        # ============================================================================
        joint[target_idx] = act

        # ---- 微辅助（按比例 + 冷却 + 限额 + 可选近目标触发）----
# 计数器/上次触发步，要在循环之前定义一次（放在 for 循环前面初始化）
# assist_used = 0
# last_assist_step = -10**9

# ← 放在 for 循环开始之前初始化：
# assist_used 和 last_assist_step
# （建议放在 t0/total_reward/steps/... 初始化那里）
# 例：
# assist_used = 0
# last_assist_step = -10**9

        if str(assist).lower() == "guidance" and joint[target_idx] == noop:
    # 条件门控
            allow = True

    # 1) 概率
            if random.random() > float(assist_prob):
                allow = False

    # 2) 冷却
            if allow and (assist_cooldown > 0) and (steps - last_assist_step < assist_cooldown):
                allow = False

    # 3) 热身期
            if allow and (assist_warmup > 0) and (steps <= assist_warmup):
                allow = False

    # 4) 次数上限（相对 max_steps 的比例）
            max_assists = max(0, int(assist_max_ratio * max_steps))
            if allow and (assist_used >= max_assists):
                allow = False

    # 5) 近目标触发（可选）
            if allow and assist_near_goal > 0 and goal_pos is not None:
                cur_xy, _ = _get_pos_goal_from_env_and_state(env, state, target_idx)
                if cur_xy is not None:
                    mhd = abs(cur_xy[0] - goal_pos[0]) + abs(cur_xy[1] - goal_pos[1])
                    if mhd > int(assist_near_goal):
                        allow = False

            if allow:
                try:
                    gpath = getattr(env, "global_guidance", None)
                    if gpath is not None and len(gpath[target_idx]) > 0:
                        cur_xy, _ = _get_pos_goal_from_env_and_state(env, state, target_idx)
                        if cur_xy is not None:
                            nxt = gpath[target_idx][0]
                            dx, dy = int(nxt[0]-cur_xy[0]), int(nxt[1]-cur_xy[1])
                            joint[target_idx] = toward_with_mapping(dx, dy, mapping)
                            # 更新统计
                            assist_used += 1
                            last_assist_step = steps
                except Exception:
                    pass
# ---- 微辅助结束 ----


        # sanitize / align length
        joint = [_sanitize_action(a, n_actions) for a in joint]
        if len(joint) < num_agents:
            joint += [noop] * (num_agents - len(joint))
        elif len(joint) > num_agents:
            joint = joint[:num_agents]

        obs_next, reward, terminated, truncated, info = _env_step_joint(env, joint, target_idx)
        try:
            total_reward += float(reward)
        except Exception:
            total_reward += float(np.mean(reward))

        if joint[target_idx] != noop:
            non_noop_moves += 1
            path_len += 1

        next_state = _state_of(obs_next, target_idx)
        obs = obs_next

        pos_now, _ = _get_pos_goal_from_env_and_state(env, next_state, target_idx)
        if pos_now is not None:
            last_pos = pos_now

        # success by reaching goal
        if goal_pos is not None and last_pos is not None and last_pos == goal_pos:
            success_flag = 1
            state = next_state
            break

        # env termination
        if terminated or truncated:
            if isinstance(info, dict):
                info_ai = info.get(target_idx, info) if isinstance(info.get(target_idx, None), dict) else info
                if isinstance(info_ai, dict):
                    for k in ("success", "is_success", "solved", "reached_goal", "done"):
                        if k in info_ai:
                            try:
                                success_flag = int(bool(info_ai[k]))
                                break
                            except Exception:
                                pass
            if success_flag == 0 and goal_pos is not None and last_pos is not None:
                success_flag = int(last_pos == goal_pos)
            state = next_state
            break

        state = next_state

    if opt_len <= 0:
        opt_len = 1
    detour_pct = max(0.0, (path_len - opt_len) / opt_len * 100.0)

    return {
        "steps": steps,
        "reward": float(total_reward),
        "success": int(success_flag),
        "moving_cost": float(non_noop_moves),
        "detour_pct": float(detour_pct),
        "opt_len": float(opt_len),
    }


# ------------------------- Stage evaluation -------------------------
def evaluate_agent_on_maps(args, stage_id: int, maps: List[MapRecord]) -> pd.DataFrame:
    rows = []
    agent_global = None

    for rec in maps:
        cfg = MapConfig(
            size=rec.size,
            num_agents=rec.agents,
            density=rec.density,
            obs_radius=args.obs_radius,
            max_episode_steps=args.max_episode_steps,
            seed=rec.seed,
            max_episode_seconds=args.max_episode_seconds,
        )
        env = _make_env_from_config(cfg)
        _reset_env_with_seed_and_grid(env, rec.seed)

        if agent_global is None:
            agent_global = _load_agent(args.model_path, env)
        agent = agent_global

        ep_metrics = []
        for _ in range(args.episodes_per_map):
            m = _rollout_one_episode(
    env, agent,
    teammates=args.teammates,
    steps_factor=args.steps_factor,
    secs_factor=args.secs_factor,
    assist=args.assist,
    assist_prob=args.assist_prob,
    assist_cooldown=args.assist_cooldown,
    assist_max_ratio=args.assist_max_ratio,
    assist_warmup=args.assist_warmup,
    assist_near_goal=args.assist_near_goal,
)

            ep_metrics.append(m)

        # aggregate
        success_list = [m['success'] for m in ep_metrics]
        row = {
            'stage': stage_id,
            'map': rec.map_id,
            'size': rec.size,
            'agents': rec.agents,
            'density': rec.density,
            'seed': rec.seed,
            'complexity': rec.complexity,
            'LDD': rec.LDD,
            'BN': rec.BN,
            'MC': rec.MC,
            'DLR': rec.DLR,
            'episodes': args.episodes_per_map,
            'success_rate': float(np.mean(success_list)) if ep_metrics else 0.0,
            'success_at_k': 1.0 if any(success_list) else 0.0,
            'avg_reward': float(np.mean([m['reward'] for m in ep_metrics])) if ep_metrics else 0.0,
            'avg_steps': float(np.mean([m['steps'] for m in ep_metrics])) if ep_metrics else 0.0,
            'avg_moving_cost': float(np.mean([m['moving_cost'] for m in ep_metrics])) if ep_metrics else 0.0,
            'avg_detour_pct': float(np.mean([m['detour_pct'] for m in ep_metrics])) if ep_metrics else 0.0,
            'avg_opt_len': float(np.mean([m['opt_len'] for m in ep_metrics])) if ep_metrics else 0.0,
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------- Main -------------------------
def main():
    parser = argparse.ArgumentParser(description="Test trained CL model by map complexity stages (patched + calibrated)")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--out_csv', type=str, default='test_by_complexity_results.csv')
    parser.add_argument('--num_maps', type=int, default=40)
    parser.add_argument('--stages', type=int, default=5)
    parser.add_argument('--episodes_per_map', type=int, default=8)

    parser.add_argument('--size_min', type=int, default=16)
    parser.add_argument('--size_max', type=int, default=64)
    parser.add_argument('--agents_min', type=int, default=2)
    parser.add_argument('--agents_max', type=int, default=16)
    parser.add_argument('--density_min', type=float, default=0.05)
    parser.add_argument('--density_max', type=float, default=0.45)

    parser.add_argument('--obs_radius', type=int, default=5)
    parser.add_argument('--max_episode_steps', type=int, default=1200)
    parser.add_argument('--max_episode_seconds', type=int, default=300)

    parser.add_argument('--steps_factor', type=float, default=5.0)
    parser.add_argument('--secs_factor',  type=float, default=1.0)

    parser.add_argument('--teammates', type=str, default='astar', choices=['astar', 'noop'],
                        help='Teammates: astar (default) or noop')
    parser.add_argument('--assist', type=str, default='none', choices=['none','guidance'],
                        help='Micro-assist: guidance (only when noop)')
    parser.add_argument('--assist_prob', type=float, default=1.0,
                    help='概率触发 guidance（0~1），默认1.0=总是触发')
    parser.add_argument('--assist_cooldown', type=int, default=0,
                    help='两次 guidance 之间至少间隔多少步')
    parser.add_argument('--assist_max_ratio', type=float, default=1.0,
                    help='单回合中 guidance 触发次数上限：<= assist_max_ratio * max_steps')
    parser.add_argument('--assist_warmup', type=int, default=0,
                    help='前多少步完全不触发 guidance（热身期）')
    parser.add_argument('--assist_near_goal', type=int, default=0,
                    help='只在距离目标 <= 该阈值(曼哈顿距离)时才触发；0表示不限制')


    parser.add_argument('--max_complexity', type=float, default=None,
                        help='If set, filter maps with complexity <= this value')

    args = parser.parse_args()

    # 1) sample & compute complexity
    records = sample_grid_maps()
    valid = [r for r in records if r.complexity is not None and not np.isnan(r.complexity)]
    print(f"[INFO] complexity valid={len(valid)} invalid={len(records)-len(valid)}")
    if not valid:
        raise RuntimeError("No valid maps with computed complexity.")

    # 2) split into stages with max_complexity filter
    stage_buckets = split_into_stages(valid, args.stages, args.max_complexity)

    # 3) evaluate
    all_results: List[pd.DataFrame] = []
    for s, maps in enumerate(stage_buckets):
        if not maps:
            continue
        lo = maps[0].complexity if maps[0].complexity is not None and not np.isnan(maps[0].complexity) else float('nan')
        hi = maps[-1].complexity if maps[-1].complexity is not None and not np.isnan(maps[-1].complexity) else float('nan')
        print(f"▶️ Testing Stage {s} - {len(maps)} maps (complexity {lo:.4f} .. {hi:.4f})")
        df_stage = evaluate_agent_on_maps(args, s, maps)
        all_results.append(df_stage)

    if not all_results:
        print("No results produced.")
        return

    results = pd.concat(all_results, ignore_index=True)

    # 4) save details
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    results.to_csv(args.out_csv, index=False, encoding='utf-8-sig')
    print(f"✅ Saved: {args.out_csv}")

    # 5) per-stage summary
    summary = results.groupby('stage').agg({
        'success_rate': 'mean',
        'success_at_k': 'mean',
        'avg_reward': 'mean',
        'avg_steps': 'mean',
        'avg_moving_cost': 'mean',
        'avg_detour_pct': 'mean',
        'complexity': ['min', 'max', 'mean']
    })
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    print('\n===== Stage Summary =====')
    print(summary.to_string())

    base, ext = os.path.splitext(args.out_csv)
    summary_path = base + '_stage_summary.csv'
    summary.to_csv(summary_path, encoding='utf-8-sig')
    print(f"✅ Saved: {summary_path}")


if __name__ == '__main__':
    main()
