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

# ===================== 配置区（按需修改） =====================
# 1) YAML 地图配置
MAP_SETTINGS_PATH = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main - train/g2rl/map_settings_generated_new.yaml"

# 2) 复杂度 CSV（由 infer_complexity.py 生成）
COMPLEXITY_CSV = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main-nn/0827result/maps_features_with_complexity.csv"

# 3) 复杂度CSV里用于匹配地图ID的列（csv）与 YAML 的对应字段（yaml）
#    例如：CSV 里是 config_id，YAML 里是 name，则对应如下：
MAP_ID_COL_IN_CSV  = "config_id"   # 也可换成 "grid_hash"
MAP_ID_COL_IN_YAML = "name"        # YAML 里地图的唯一标识字段

# 4) 训练超参
N_STAGES = 5
MIN_PER_STAGE = 10
MIN_EPISODES_PER_STAGE = 300
THRESHOLD = 0.80
WINDOW_SIZE = 100
NUM_EPISODES = 300
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 500
DECAY_RANGE = 10_000
MAX_EPISODE_SECONDS = 30
LOG_DIR = "logs"
RUN_DIR = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main/final_trainig_1"
MODEL_OUT = "models/best_model.pt"
# ============================================================


# -------- 查找观测中的数组，并统一到 [C, D, H, W] --------
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
            arr = _find_array(v); 
            if arr is not None: 
                return arr
        return None
    if isinstance(obj, dict):
        preferred = ("obs","observation","view","state","tensor","grid","image","local_obs","global_obs")
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
    """
    从任意结构的单个智能体观测 s 构造 [1, C, D, H, W] 的 float32 tensor，
    并把“通道维”对齐到 expected_c：
      - C==expected_c：原样
      - C==1 且 expected_c>1：repeat 到 expected_c
      - 1<C<expected_c：零填充到 expected_c
      - C>expected_c：裁剪前 expected_c 个通道
    """
    import numpy as np
    import torch

    arr = _find_array(s)
    if arr is None:
        raise ValueError("无法从观测中提取数组/张量。")

    arr = np.array(arr)

    # 先归一到 4 维 [C,D,H,W]
    if arr.ndim == 5:          # [N,?, ?, ?, ?]
        if arr.shape[1] == expected_c:
            arr = arr[0]
        elif arr.shape[-1] == expected_c:
            arr = np.transpose(arr, (0,4,1,2,3))[0]
        else:
            arr = arr[0]
    if arr.ndim == 4:          # [C,D,H,W] / [D,H,W,C] / [D,C,H,W]
        if arr.shape[0] == expected_c:
            pass
        elif arr.shape[-1] == expected_c:
            arr = np.transpose(arr, (3,0,1,2))     # DHWC -> CDHW
        elif arr.shape[1] == expected_c:
            arr = np.transpose(arr, (1,0,2,3))     # DCHW -> CDHW
    elif arr.ndim == 3:        # [D,H,W]
        arr = arr[None, ...]
    elif arr.ndim == 2:        # [H,W]
        arr = arr[None, None, ...]
    else:
        raise ValueError(f"观测维度不支持：shape={arr.shape}, ndim={arr.ndim}")

    if arr.ndim != 4:
        raise ValueError(f"预处理后不是 4 维 CDHW：shape={arr.shape}")

    C, D, H, W = arr.shape

    # 通道修正为 expected_c
    if C == expected_c:
        arr_fixed = arr
    elif C == 1 and expected_c > 1:
        arr_fixed = np.repeat(arr, expected_c, axis=0)
    elif C < expected_c:
        pad = np.zeros((expected_c - C, D, H, W), dtype=arr.dtype)
        arr_fixed = np.concatenate([arr, pad], axis=0)
    else:  # C > expected_c
        arr_fixed = arr[:expected_c]

    x = torch.tensor(arr_fixed[None, ...], dtype=torch.float32, device=device)  # [1,C,D,H,W]
    return x


# ============ 项目根路径（确保能 import g2rl.*） ============
project_root = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main-nn"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============ 项目模块 ============ (要求你的项目中存在这些模块)
from g2rl.environment import G2RLEnv
from g2rl.agent import DDQNAgent
from g2rl.network import CRNNModel

# ============ 安全构造器：只传 __init__ 支持的参数 ============

from types import SimpleNamespace

def _safe_get_action_space(env):
    """
    尽量稳健地拿到动作空间：
    1) 优先用 env.action_space.n
    2) 再试 env.get_action_space() 的返回
    3) 再试 env.actions 的长度
    4) 如果还没有，先 reset 一次再重复 1-3
    5) 最后兜底返回 5（常见的 stay+UDLR）
    """
    # 1) gym 风格
    asp = getattr(env, "action_space", None)
    if asp is not None and hasattr(asp, "n"):
        return asp

    # 2) 项目自定义
    gas = getattr(env, "get_action_space", None)
    if callable(gas):
        try:
            out = gas()
            if hasattr(out, "n"):
                return out
            if isinstance(out, (list, tuple)):
                return SimpleNamespace(n=len(out))
            if isinstance(out, int) and out > 0:
                return SimpleNamespace(n=out)
        except Exception:
            pass

    # 3) 直接看 env.actions
    if hasattr(env, "actions"):
        try:
            return SimpleNamespace(n=len(env.actions))
        except Exception:
            pass

    # 4) 先 reset 再试一轮
    try:
        env.reset()
    except Exception:
        pass

    asp = getattr(env, "action_space", None)
    if asp is not None and hasattr(asp, "n"):
        return asp
    gas = getattr(env, "get_action_space", None)
    if callable(gas):
        try:
            out = gas()
            if hasattr(out, "n"):
                return out
            if isinstance(out, (list, tuple)):
                return SimpleNamespace(n=len(out))
            if isinstance(out, int) and out > 0:
                return SimpleNamespace(n=out)
        except Exception:
            pass
    if hasattr(env, "actions"):
        try:
            return SimpleNamespace(n=len(env.actions))
        except Exception:
            pass

    # 5) 兜底
    return SimpleNamespace(n=5)



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

    # 将 grid/starts/goals 等作为属性挂载（如果 YAML 有这些）
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

# ================== 用 CSV 合并复杂度 ==================
def _merge_complexity_from_csv(base_map_settings: Dict[str, dict]) -> Dict[str, dict]:
    """
    稳健合并 nn_complexity：
    1) 自动寻找 CSV 与 YAML 的共同 ID 列（优先：config_id、grid_hash、name、map_id）
    2) 若无共同 ID，则使用复合键 (size,num_agents,density,obs_radius,max_episode_steps) 匹配
       - density 等浮点列会做四舍五入以避免微小数值误差
    3) 对同一 ID 的多行取均值（需要的话可以改成取最后一条）
    4) 打印匹配统计
    """
    import numpy as np
    import pandas as pd
    import os

    if not os.path.exists(COMPLEXITY_CSV):
        raise FileNotFoundError(f"未找到复杂度CSV：{COMPLEXITY_CSV}")

    df = pd.read_csv(COMPLEXITY_CSV)

    # 兜底：没有 nn_complexity 就用 1 - nn_pred_success_rate
    if "nn_complexity" not in df.columns:
        if "nn_pred_success_rate" in df.columns:
            df["nn_complexity"] = 1.0 - pd.to_numeric(df["nn_pred_success_rate"], errors="coerce")
        else:
            raise ValueError("复杂度CSV缺少 nn_complexity 且没有 nn_pred_success_rate 无法推导。")

    # 规范成数值
    df["nn_complexity"] = pd.to_numeric(df["nn_complexity"], errors="coerce")

    # ===== 1) 寻找共同 ID 列 =====
    csv_id_candidates  = [c for c in ["config_id", "grid_hash", "name", "map_id"] if c in df.columns]
    yaml_id_candidates = set()
    for spec in base_map_settings.values():
        yaml_id_candidates.update(spec.keys())
    yaml_id_candidates = [c for c in ["config_id", "grid_hash", "name", "map_id"] if c in yaml_id_candidates]

    # 取第一个共同列作为主键
    common_ids = [c for c in csv_id_candidates if c in yaml_id_candidates]
    chosen_id = common_ids[0] if common_ids else None

    # ===== 2) 如果没有共同 ID，用复合键匹配 =====
    use_composite = (chosen_id is None)

    # 预处理 DataFrame（按 chosen_id 或复合键聚合）
    if not use_composite:
        # 同一 ID 多行 -> 均值
        agg = (
            df.groupby(chosen_id, dropna=False)["nn_complexity"]
              .mean()
              .reset_index()
              .rename(columns={"nn_complexity": "complexity"})
        )
        # 映射：id -> complexity
        comp_map = dict(zip(agg[chosen_id].astype(str), agg["complexity"].astype(float)))
    else:
        # 复合键：size,num_agents,density,obs_radius,max_episode_steps
        needed = ["size", "num_agents", "density", "obs_radius", "max_episode_steps"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"无法用复合键匹配，CSV 缺少列：{missing}")

        # 数值规范：四舍五入避免浮点误差
        df["_size"] = pd.to_numeric(df["size"], errors="coerce").astype("Int64")
        df["_nag"]  = pd.to_numeric(df["num_agents"], errors="coerce").astype("Int64")
        df["_den"]  = pd.to_numeric(df["density"], errors="coerce").round(4)
        df["_obs"]  = pd.to_numeric(df["obs_radius"], errors="coerce").astype("Int64")
        df["_mep"]  = pd.to_numeric(df["max_episode_steps"], errors="coerce").astype("Int64")

        comp = (
            df.dropna(subset=["_size","_nag","_den","_obs","_mep"])
              .groupby(["_size","_nag","_den","_obs","_mep"])["nn_complexity"]
              .mean()
              .reset_index()
              .rename(columns={"nn_complexity":"complexity"})
        )
        # 映射：tuple -> complexity
        comp_map = { (int(r["_size"]), int(r["_nag"]), float(r["_den"]), int(r["_obs"]), int(r["_mep"])): float(r["complexity"])
                     for _, r in comp.iterrows() }

    # ===== 3) 把 complexity 写回 spec =====
    matched, unmatched = 0, 0
    out: Dict[str, dict] = {}

    for name, spec in base_map_settings.items():
        new_spec = dict(spec)
        cpx_val = np.nan

        if not use_composite:
            # 主键匹配
            key = spec.get(chosen_id, None)
            if key is None and chosen_id == "name":
                # 退化：用 dict 的 key（name）当作 name
                key = name
            if key is not None:
                cpx_val = comp_map.get(str(key), np.nan)
        else:
            # 复合键匹配
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

    print(f"🧩 复杂度匹配统计：matched={matched}, unmatched={unmatched} "
          f"| 模式={'复合键' if use_composite else f'ID列({chosen_id})'}")

    if unmatched > 0:
        # 打印前几个未匹配样本，便于你确认 ID 是否一致
        examples = [k for k,v in out.items() if not np.isfinite(v.get('complexity', np.nan))][:5]
        print(f"⚠️ 有 {unmatched} 张地图未匹配到复杂度（未参与量化）。示例：{examples}")

    return out


# ================== 阶段切分（按复杂度分位） ==================
def _build_stages_by_quantile_df(df: pd.DataFrame, n_stages: int = 5, min_per_stage: int = 5):
    df = df.dropna(subset=["complexity"]).copy()
    if len(df) == 0:
        raise ValueError("没有可用地图用于 complexity 课程（complexity 全 NaN）。")
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

# ================== Scheduler ==================
class ComplexityScheduler:
    """
    - 每阶段持有一组 maps（由 complexity 分位切分）
    - 每阶段至少跑 min_episodes_per_stage 集
    - 达到阈值后晋级
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
                 seed: int = 0):
        self.min_episodes_per_stage = int(min_episodes_per_stage)
        self.threshold = float(threshold)
        self.window_size = int(window_size)
        self.use_window_sr = bool(use_window_sr)

        # 构造包含 complexity 的 DataFrame
        rows = []
        for name, spec in base_map_settings.items():
            rows.append({"name": name, "complexity": spec.get("complexity", np.nan), "spec": spec})
        df = pd.DataFrame(rows)
        stages = _build_stages_by_quantile_df(df, n_stages=n_stages, min_per_stage=min_per_stage)

        self._rng = random.Random(seed)
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

# ================== 训练 ==================
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

    # 输出目录
    if run_dir is None:
        run_dir = Path(log_dir) / get_timestamp()
    else:
        run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    # 初始化第一个 env 以获取动作空间
    from types import SimpleNamespace

    first_name = next(iter(map_settings))
    first_env = build_env_from_raw(map_settings[first_name])
    # ==== 已有：first_env 构建 & 重置 ====
    try:
        first_env.reset()
    except Exception:
        pass

# ==== 推断 n_actions ====
    def _infer_n_actions(env) -> int:
        asp = getattr(env, "action_space", None)
        if asp is not None and hasattr(asp, "n"):
            return int(asp.n)
        if hasattr(env, "get_action_space") and callable(env.get_action_space):
            out = env.get_action_space()
            if hasattr(out, "n"):
                return int(out.n)
            if hasattr(out, "__len__"):
                return len(out)
            if isinstance(out, int):
                return out
        if hasattr(env, "actions") and env.actions is not None:
            return len(env.actions)
        return 5

    n_actions = _infer_n_actions(first_env)
    action_space_list = list(range(n_actions))
    print(f"✅ n_actions = {n_actions}")

# ==== 确定/探测通道数（你也可以写死 11）====
# 如果你之前已经通过探针拿到了 in_channels，就用那个；否则先取一个观测探测
    probe_name = next(iter(map_settings))
    probe_env  = build_env_from_raw(map_settings[probe_name])
    obs0, _    = probe_env.reset()
    state0     = obs0[0] if isinstance(obs0, (list, tuple)) else obs0

# 这里可直接固定：in_channels = 11
# 或者如果你已经实现了自动探测，就替换成探测结果
    in_channels = 11

# ==== 统一预处理函数 ====
    def preprocess_batch1(s):
    # 返回 [1, C, D, H, W]，并强制 C = in_channels
        return _obs_to_tensor_CDHW(s, device, expected_c=in_channels)

    def preprocess_single(s):
    # 返回 [C, D, H, W]（存经验/拼 batch 用）
        return preprocess_batch1(s).squeeze(0)

# ==== 预热 LazyConv3d（必须在任何 store()/retrain() 之前）====
    with torch.no_grad():
        _ = model(preprocess_batch1(state0))
    print(f"🔥 warmed model with in_channels={in_channels}")

# ==== 初始化 Agent（注意：前三个必填参数顺序：q_network, model, action_space）====
    agent = DDQNAgent(
    model,                 # q_network
    model,                 # model（与你的签名一致，两个都传同一个实例）
    action_space_list,     # action_space: [0..n_actions-1]
    lr=lr,
    decay_range=decay_range,
    device=device,
    replay_buffer_size=replay_buffer_size,
    # 你的类若支持这个关键字就传；否则删掉
    obs_preprocessor=preprocess_single,
    )

# 统一预处理：和你做动作选择时同一函数
    agent.obs_preprocessor = lambda s: _obs_to_tensor_CDHW(s, device, expected_c=in_channels)




    training_logs = []
    pbar = tqdm(range(num_episodes), desc='Episodes', dynamic_ncols=True)

    episode = 0
    success_count_total = 0

    while (scheduler is None) or (scheduler.current_stage <= scheduler.max_stage):
        if episode >= num_episodes:
            break

        # 1) 当前阶段地图
        cur_map_cfg = scheduler.get_updated_map_settings() if scheduler else map_settings
        map_type, cfg = next(iter(cur_map_cfg.items()))
        env = build_env_from_raw(cfg)

        stage_id = (scheduler.current_stage if scheduler else -1)
        cpx_val = cfg.get("complexity", None)
        if cpx_val is not None and np.isfinite(cpx_val):
            pbar.write(f"🟢 地图：{map_type} | Stage {stage_id} | Agents={env.num_agents} | Complexity={float(cpx_val):.3f}")
        else:
            pbar.write(f"🟢 地图：{map_type} | Stage {stage_id} | Agents={env.num_agents}")

        # 2) reset
        try:
            obs, info = env.reset()
        except Exception:
            obs = env.reset()
            info = {}

        # 目标代理（其它用 A* 做队友）
        target_idx = np.random.randint(env.num_agents)
        teammates = [AStarAgent() if i != target_idx else None for i in range(env.num_agents)]
        goal = tuple(env.goals[target_idx])
        state = obs[target_idx]

        # 3) 一集
        success_flag = False
        retrain_count = 0
        episode_start_time = time.time()

        # 你原本的步数增长策略
        timesteps_per_episode = 50 + 10 * episode

        for t in range(timesteps_per_episode):
            if time.time() - episode_start_time > max_episode_seconds:
                pbar.write(f"⏰ Episode {episode} 超时（>{max_episode_seconds}s）")
                break

            # ===== 替换开始：我们智能体的动作选择（确保喂给网络的是 [1, in_channels, D, H, W]） =====
            actions = []
            for i in range(env.num_agents):
                if i == target_idx:
        # 关键：统一观测到 [1, in_channels, D, H, W]
                    x = _obs_to_tensor_CDHW(obs[i], device, expected_c=in_channels)

        # 一次性防御性检查（推荐保留，定位维度问题非常有用）
                    if x.shape[1] != in_channels:
                        raise RuntimeError(f"[PrepMismatch] x.shape={tuple(x.shape)}, expected in_channels={in_channels}")

                    with torch.no_grad():
                        q = agent.q_network(x)
                        a = int(torch.argmax(q, dim=1).item())
                    actions.append(a)
                else:
                    try:
                        actions.append(int(teammates[i].act(obs[i])))
                    except Exception:
                        actions.append(0)
# ===== 替换结束 =====



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

        # 4) 统计与日志
        if success_flag:
            success_count_total += 1

        episode += 1
        success_rate = success_count_total / max(1, episode)
        writer.add_scalar('Success/episode', int(success_flag), episode)
        writer.add_scalar('SuccessRate/global', success_rate, episode)

        pbar.set_postfix(
            Stage=(stage_id if scheduler else "-"),
            success=int(success_flag),
            SR=f"{success_rate:.3f}",
        )
        pbar.update(1)  # ← 别忘了推进进度条

        training_logs.append({
            "episode": episode,
            "stage": stage_id,
            "map": map_type,
            "agents": getattr(env, "num_agents", None),
            "complexity": (float(cpx_val) if cpx_val is not None else np.nan),
            "success": int(success_flag),
            "success_rate": float(success_rate),
        })

        # 5) 课程逻辑
        if scheduler is not None:
            scheduler.add_episode_result(int(success_flag))
            if scheduler.should_advance():
                scheduler.advance(pbar)
                if scheduler.is_done():
                    break
            else:
                if scheduler._ep_in_stage >= scheduler.min_episodes_per_stage:
                    scheduler.repeat_stage(pbar)

    # 保存日志
    df = pd.DataFrame(training_logs)
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "episodes.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"📝 每集日志已保存：{csv_path}")

    writer.close()
    return agent

def _find_array(obj):
    """在 obj(可能是dict/list/tuple/ndarray/torch.Tensor) 中递归寻找第一个 ndarray 或可转 ndarray 的对象。"""
    import numpy as np
    try:
        import torch
        is_tensor = True
    except Exception:
        is_tensor = False

    # 直接是 ndarray
    if isinstance(obj, np.ndarray):
        return obj
    # torch tensor
    if is_tensor and hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        try:
            return obj.detach().cpu().numpy()
        except Exception:
            pass
    # 列表/元组：依次找
    if isinstance(obj, (list, tuple)):
        for v in obj:
            arr = _find_array(v)
            if arr is not None:
                return arr
        return None
    # 字典：优先常用键，再全量遍历
    if isinstance(obj, dict):
        preferred = ("obs", "observation", "view", "state", "tensor", "grid", "image", "local_obs", "global_obs")
        for k in preferred:
            if k in obj:
                arr = _find_array(obj[k])
                if arr is not None:
                    return arr
        # 退化：遍历所有 value
        for v in obj.values():
            arr = _find_array(v)
            if arr is not None:
                return arr
        return None

    # 标量/其他：尝试直接转
    try:
        arr = np.array(obj)
        if arr.ndim > 0:  # 至少 1 维才算“像数组”
            return arr
    except Exception:
        pass
    return None


def _to_CDHW(arr):
    """把 ndarray 统一成 [C, D, H, W]。允许输入维度为 2/3/4/5。"""
    import numpy as np
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    if arr.ndim == 5:
        # 可能是 [N,C,D,H,W] 或 [N,D,H,W,C]
        if arr.shape[1] in (1,2,3,4,5,8,11,16):  # 看第2维像“通道”
            arr = arr[0]  # -> [C,D,H,W]
        else:
            # 假定是 [N,D,H,W,C]
            arr = np.transpose(arr, (0,4,1,2,3))[0]  # -> [C,D,H,W]
    elif arr.ndim == 4:
        # 可能是 [C,D,H,W] 或 [D,H,W,C]
        if arr.shape[0] in (1,2,3,4,5,8,11,16):
            pass  # 已是 [C,D,H,W]
        elif arr.shape[-1] in (1,2,3,4,5,8,11,16):
            arr = np.transpose(arr, (3,0,1,2))  # DHWC -> CDHW
        else:
            # 不确定，把最后一维当通道：H W C D -> C H W D 再调回 C D H W（极少见）
            arr = np.transpose(arr, (3,0,1,2))
    elif arr.ndim == 3:
        # [D,H,W] -> [1,D,H,W]
        arr = arr[None, ...]
    elif arr.ndim == 2:
        # [H,W] -> [1,1,H,W]
        arr = arr[None, None, ...]
    else:
        raise ValueError(f"无法归一到 [C,D,H,W]：arr.ndim={arr.ndim}, shape={arr.shape}")
    return arr

# ================== 主程序 ==================
if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 1) 读 YAML
    with open(MAP_SETTINGS_PATH, "r", encoding="utf-8") as f:
        base_map_settings = yaml.safe_load(f)

    if isinstance(base_map_settings, list):
        base_map_settings = {(m.get("name") or f"map_{i}"): m for i, m in enumerate(base_map_settings)}

    # 2) 合并复杂度（来自 CSV 的 nn_complexity）
    base_map_settings = _merge_complexity_from_csv(base_map_settings)

    # 3) 构建 Scheduler
    scheduler = ComplexityScheduler(
        base_map_settings=base_map_settings,
        n_stages=N_STAGES,
        min_per_stage=MIN_PER_STAGE,
        min_episodes_per_stage=MIN_EPISODES_PER_STAGE,
        threshold=THRESHOLD,
        window_size=WINDOW_SIZE,
        use_window_sr=True,
        shuffle_each_stage=True,
        seed=0,
    )



    # 在创建 scheduler 之后、创建 model 之前：
    first_name = next(iter(base_map_settings))
    probe_env = build_env_from_raw(base_map_settings[first_name])
    # ---------- 观测提取与整形工具 ----------
    probe_name = next(iter(base_map_settings))
    probe_env  = build_env_from_raw(base_map_settings[probe_name])
    obs_info   = probe_env.reset()
# 兼容 (obs, info) 或仅 obs
    if isinstance(obs_info, tuple) and len(obs_info) >= 1:
        obs = obs_info[0]
    else:
        obs = obs_info

    state0 = obs[0] if isinstance(obs, (list, tuple)) and len(obs) > 0 else obs

    arr_raw = _find_array(state0)
    if arr_raw is None:
    # 打印可用键帮助定位
        if isinstance(state0, dict):
            print("❌ 在 state0 中未找到可用数组。state0.keys():", list(state0.keys()))
        else:
            print("❌ 在 state0 中未找到可用数组。type(state0):", type(state0))
        raise ValueError("无法从观测中提取张量，请告诉我 state 结构或贴一段样例，我来适配。")

    arr_cdwh = _to_CDHW(arr_raw)     # -> [C,D,H,W]
    in_channels = 11
    print(f"🧪 推断观测通道数 in_channels={in_channels}, sample shape={arr_cdwh.shape}")




# 注意：DDQNAgent 里通常会从 action_space.n 推断 num_actions，
# 如果需要传入，也用我们上一步推断的动作数 n_actions。


    # 4) 模型与设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNNModel().to(device)
    # 预热 LazyConv3d，让它以你期望的通道数定型（in_channels）
    x0 = _obs_to_tensor_CDHW(state0, device, expected_c=in_channels)  # [1,C,D,H,W]
    with torch.no_grad():
        _ = model(x0)  # 首次前向，LazyConv3d 固定 in_channels=C


    # 5) 训练
    agent = train(
        model=model,
        scheduler=scheduler,
        map_settings=scheduler.get_updated_map_settings(),  # 初次取一张；train 内每集会再取
        map_probs=None,
        num_episodes=NUM_EPISODES,
        batch_size=BATCH_SIZE,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        decay_range=DECAY_RANGE,
        log_dir=LOG_DIR,
        device=device,
        max_episode_seconds=MAX_EPISODE_SECONDS,
        run_dir=RUN_DIR,
    )

    # 6) 保存模型权重
    out_path = Path(MODEL_OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path.as_posix())
    print(f"✅ 模型已保存到 {out_path}")
