# # # g2rl/map_settings.py

# from itertools import product
# import yaml
# import os




# def generate_map_settings(map_sizes, agent_counts, densities):
#     combinations = list(product(map_sizes, agent_counts, densities))
#     settings = {}
#     for i, (size, agents, density) in enumerate(combinations):
#         settings[f"map_{i}"] = {
#             "size": size,
#             "num_agents": agents,
#             "density": density,
#             "obs_radius": 5,
#             "max_episode_steps": 100
#         }
#     return settings

# if __name__ == '__main__':
#     map_sizes = [32, 64, 96, 128]
#     agent_counts = [2, 4, 8, 16]
#     densities = [0.1, 0.3, 0.5, 0.7]

#     # ✅ 生成地图设置
#     settings = generate_map_settings(map_sizes, agent_counts, densities)
#     # ✅ 确保输出目录存在
#     os.makedirs("g2rl", exist_ok=True)

#     # ✅ 保存为 YAML 文件
#     with open("g2rl/map_settings_generated.yaml", "w", encoding="utf-8") as f:
#         yaml.dump(settings, f, allow_unicode=True)

#     print(f"✅ 已生成 {len(settings)} 个地图配置，保存至 g2rl/map_settings_generated.yaml")


#使用模型评估地图complexity


# import joblib
# import pandas as pd
# from g2rl.environment import G2RLEnv

# def predict_complexity(config_or_env, model_path="complexity_model.pkl") -> float:
#     """
#     输入地图配置（dict）或 G2RLEnv 环境对象，预测其复杂度。

#     参数:
#         config_or_env (dict or G2RLEnv): 地图配置字典 或 已初始化的 G2RLEnv 对象
#         model_path (str): 保存的模型路径

#     返回:
#         float: 预测的 complexity 分数
#     """
#     # 加载模型
#     try:
#         model = joblib.load(model_path)
#     except Exception as e:
#         raise RuntimeError(f"❌ 无法加载模型文件: {model_path}, 错误: {e}")

#     # 提取特征（确保字段顺序与训练时一致）
#     if isinstance(config_or_env, G2RLEnv):
#         size = config_or_env.grid_config.size
#         agents = config_or_env.num_agents
#         density = config_or_env.grid_config.density
#         bottleneck = getattr(config_or_env, 'bottleneck_score', 0.0)
#         goal_dist = getattr(config_or_env, 'avg_goal_dist', 0.0)
#         collision_risk = getattr(config_or_env, 'collision_risk', 0.0)
#     elif isinstance(config_or_env, dict):
#         size = config_or_env.get('size', 16)
#         agents = config_or_env.get('num_agents', 4)
#         density = config_or_env.get('density', 0.1)
#         bottleneck = config_or_env.get('bottleneck_score', 0.0)
#         goal_dist = config_or_env.get('avg_goal_dist', 0.0)
#         collision_risk = config_or_env.get('collision_risk', 0.0)
#     else:
#         raise ValueError("❌ 输入必须是 G2RLEnv 或 dict 类型")

#     # 构造特征输入
#     features = pd.DataFrame([{
#         'Size': size,
#         'Agents': agents,
#         'Density': density,
#         'Bottleneck': bottleneck,
#         'GoalDist': goal_dist,
#         'CollisionRisk': collision_risk
#     }])

#     # 模型预测
#     complexity = model.predict(features)[0]
#     return complexity

# generate_maps_with_grid_only.py# generate_maps_with_grid_only.py
from itertools import product
import os
import yaml
import numpy as np
from typing import List, Tuple, Optional
# g2rl/map_settings.py
import os
import json
from typing import Dict, Any, Optional
import numpy as np

# 依赖你已有的 complexity.py
from g2rl.complexity import compute_complexity

# 多项式权重文件（如果训练过，会放在 g2rl/ 目录）
_POLY_JSON_PATH = os.path.join(os.path.dirname(__file__), "weights_poly.json")
_BASE_ORDER = ["Size", "Agents", "Density", "LDD", "BN", "MC", "DLR"]

# 懒加载缓存
__poly_loaded = False
__poly_terms = None            # list[{"name": str, "coef": float}]
__poly_intercept = 0.0
__poly_degree = 2
__poly_feature_names = None    # list[str]
__poly_available = False       # 是否可用（文件存在且依赖可用）


def __load_poly_model():
    """尝试读取 weights_poly.json 并准备特征名；失败则保持回退模式。"""
    global __poly_loaded, __poly_terms, __poly_intercept, __poly_degree
    global __poly_feature_names, __poly_available

    if __poly_loaded:
        return

    __poly_loaded = True
    __poly_available = False

    if not os.path.exists(_POLY_JSON_PATH):
        return  # 没有文件，直接回退

    # 尝试懒加载 sklearn，仅在需要时导入
    try:
        from sklearn.preprocessing import PolynomialFeatures  # type: ignore
    except Exception:
        return  # 没装 sklearn，就回退

    try:
        with open(_POLY_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        __poly_degree = int(data.get("poly_degree", 2))
        __poly_terms = data.get("terms", []) or []
        __poly_intercept = float(data.get("intercept", 0.0))

        # 仅用于生成稳定的项名称顺序（不需要真实数据）
        pf = PolynomialFeatures(degree=__poly_degree, include_bias=False)
        pf.fit(np.zeros((1, len(_BASE_ORDER))))
        __poly_feature_names = list(pf.get_feature_names_out(_BASE_ORDER))

        # 只有当 terms 非空时才视为可用
        __poly_available = len(__poly_terms) > 0

    except Exception:
        # 任意异常都回退到默认 Complexity
        __poly_available = False
        __poly_terms = []
        __poly_intercept = 0.0
        __poly_feature_names = None


def __predict_with_poly(base_feats: Dict[str, float]) -> Optional[float]:
    """使用已加载的多项式公式计算评分；若不可用返回 None。"""
    __load_poly_model()
    if not __poly_available or not __poly_terms or not __poly_feature_names:
        return None

    # 再次导入（防止上层环境热重载）
    try:
        from sklearn.preprocessing import PolynomialFeatures  # type: ignore
    except Exception:
        return None

    # 1) 按训练时顺序取基础特征
    xbase = np.array([[float(base_feats[k]) for k in _BASE_ORDER]], dtype=float)

    # 2) 生成同顺序的多项式项
    pf = PolynomialFeatures(degree=__poly_degree, include_bias=False)
    pf.fit(np.zeros((1, len(_BASE_ORDER))))
    Xp = pf.transform(xbase)
    names = list(pf.get_feature_names_out(_BASE_ORDER))
    name_to_idx = {n: i for i, n in enumerate(names)}

    # 3) 组装系数向量
    coef_vec = np.zeros(Xp.shape[1], dtype=float)
    for t in __poly_terms:
        n = t.get("name")
        c = float(t.get("coef", 0.0))
        idx = name_to_idx.get(n)
        if idx is not None:
            coef_vec[idx] = c

    # 4) 线性组合 + 截距
    y = float(np.dot(Xp[0], coef_vec) + __poly_intercept)
    # 如果目标是概率/成功率，通常裁剪到 [0,1]
    return float(np.clip(y, 0.0, 1.0))


def predict_complexity(cfg: Dict[str, Any],
                       connectivity: int = 4,
                       K: int = 30,
                       seed: int = 42,
                       use_poly: bool = True) -> float:
    """
    传入一张地图配置 dict（至少包含 grid/num_agents；可选 starts/goals），返回复杂度分数。
    优先使用你训练得到的“二次多项式 + 非负 LassoCV”公式（weights_poly.json）；
    若不可用或没安装 sklearn，则回退到 complexity.py 的默认线性组合分数。
    """
    grid = np.array(cfg["grid"], dtype=np.uint8)
    agents = int(cfg.get("num_agents", 2))
    starts = cfg.get("starts", None)
    goals = cfg.get("goals", None)

    # 先用 complexity 计算基础特征 & 默认 Complexity
    res = compute_complexity(
        grid=grid,
        starts=starts,
        goals=goals,
        agents=agents,
        connectivity=connectivity,
        K=K,
        random_state=seed,
        # 不在这里传线性 weights；默认内部会用自身的 DEFAULT_WEIGHTS
    )

    if use_poly:
        base = {k: float(res[k]) for k in _BASE_ORDER}
        y = __predict_with_poly(base)
        if y is not None:
            return y

    # 回退
    return float(res["Complexity"])

def gen_random_grid(size: int,
                    density: float,
                    rng: np.random.Generator,
                    border: bool = True) -> np.ndarray:
    """
    生成 size x size 的 0/1 网格。1=障碍，0=可走。
    density: 障碍比例（0~1）；默认四周加一圈墙体，避免出界。
    """
    g = (rng.random((size, size)) < float(density)).astype(np.uint8)
    if border:
        g[0, :] = 1; g[-1, :] = 1; g[:, 0] = 1; g[:, -1] = 1
    # 兜底：避免全障碍或全空（影响后续起终点采样）
    if g.mean() < 0.02:
        k = max(1, int(0.02 * size * size))
        idx = rng.choice(size * size, size=k, replace=False)
        g.flat[idx] = 1
    if g.mean() > 0.98:
        k = max(1, int(0.02 * size * size))
        idx = rng.choice(size * size, size=k, replace=False)
        g.flat[idx] = 0
    return g

def sample_starts_goals(grid: np.ndarray,
                        num_agents: int,
                        rng: np.random.Generator) -> Tuple[Optional[List[List[int]]],
                                                           Optional[List[List[int]]]]:
    """
    在可行走单元随机采样起点/终点，各 num_agents 个。
    返回：list[list[int,int]]；如果可用格不足则返回 (None, None)。
    """
    free = np.argwhere(grid == 0)
    need = 2 * num_agents
    if len(free) < max(2, need):
        return None, None
    idx = rng.choice(len(free), size=need, replace=False)
    pts = [free[i].tolist() for i in idx]   # ← list 而不是 tuple，避免 !!python/tuple
    starts = pts[:num_agents]
    goals  = pts[num_agents:]
    return starts, goals

def generate_map_settings(map_sizes,
                          agent_counts,
                          densities,
                          seed: int = 42,
                          out_dir: str = "g2rl",
                          out_name: str = "map_settings_generated.yaml"):
    """
    只生成 YAML（列表结构），每个元素包含：
      - name / size / num_agents / density / obs_radius / max_episode_steps
      - grid: 0/1 矩阵（list of list）
      - starts, goals: list of [r, c] ；可能为 None
    """
    rng = np.random.default_rng(seed)
    combinations = list(product(map_sizes, agent_counts, densities))
    maps_yaml_list = []

    os.makedirs(out_dir, exist_ok=True)

    for i, (size, agents, density) in enumerate(combinations):
        name = f"map_{i}"
        grid = gen_random_grid(size=size, density=density, rng=rng, border=True)
        starts, goals = sample_starts_goals(grid, agents, rng)

        maps_yaml_list.append({
            "name": name,
            "size": int(size),
            "num_agents": int(agents),
            "density": float(density),
            "obs_radius": 5,
            "max_episode_steps": 100,
            "grid": grid.astype(int).tolist(),
            "starts": starts,     # list[list[int,int]] 或 None
            "goals": goals        # list[list[int,int]] 或 None
        })

    yaml_path = os.path.join(out_dir, out_name)
    with open(yaml_path, "w", encoding="utf-8") as f:
        # 用 SafeDumper，避免非标准标签（如 !!python/tuple）
        yaml.dump(maps_yaml_list, f, allow_unicode=True, Dumper=yaml.SafeDumper)
    print(f"✅ 已生成 {len(maps_yaml_list)} 张地图，写入 {yaml_path}")
    return yaml_path

if __name__ == '__main__':
    map_sizes = [32, 64, 96, 128]
    agent_counts = [2, 4, 8, 16]
    densities = [0.1, 0.3, 0.5, 0.7]

    generate_map_settings(
        map_sizes=map_sizes,
        agent_counts=agent_counts,
        densities=densities,
        seed=42,
        out_dir="g2rl",
        out_name="map_settings_generated.yaml"
    )
