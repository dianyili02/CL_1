
# # generate_maps_with_grid_only.py# generate_maps_with_grid_only.py
# from itertools import product
# import os
# import yaml
# import numpy as np
# from typing import List, Tuple, Optional
# # g2rl/map_settings.py
# import os
# import json
# from typing import Dict, Any, Optional
# import numpy as np

# # 依赖你已有的 complexity.py
# from g2rl.complexity import compute_complexity

# # 多项式权重文件（如果训练过，会放在 g2rl/ 目录）
# _POLY_JSON_PATH = os.path.join(os.path.dirname(__file__), "weights_poly.json")
# _BASE_ORDER = ["Size", "Agents", "Density", "LDD", "BN", "MC", "DLR"]

# # 懒加载缓存
# __poly_loaded = False
# __poly_terms = None            # list[{"name": str, "coef": float}]
# __poly_intercept = 0.0
# __poly_degree = 2
# __poly_feature_names = None    # list[str]
# __poly_available = False       # 是否可用（文件存在且依赖可用）


# def __load_poly_model():
#     """尝试读取 weights_poly.json 并准备特征名；失败则保持回退模式。"""
#     global __poly_loaded, __poly_terms, __poly_intercept, __poly_degree
#     global __poly_feature_names, __poly_available

#     if __poly_loaded:
#         return

#     __poly_loaded = True
#     __poly_available = False

#     if not os.path.exists(_POLY_JSON_PATH):
#         return  # 没有文件，直接回退

#     # 尝试懒加载 sklearn，仅在需要时导入
#     try:
#         from sklearn.preprocessing import PolynomialFeatures  # type: ignore
#     except Exception:
#         return  # 没装 sklearn，就回退

#     try:
#         with open(_POLY_JSON_PATH, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         __poly_degree = int(data.get("poly_degree", 2))
#         __poly_terms = data.get("terms", []) or []
#         __poly_intercept = float(data.get("intercept", 0.0))

#         # 仅用于生成稳定的项名称顺序（不需要真实数据）
#         pf = PolynomialFeatures(degree=__poly_degree, include_bias=False)
#         pf.fit(np.zeros((1, len(_BASE_ORDER))))
#         __poly_feature_names = list(pf.get_feature_names_out(_BASE_ORDER))

#         # 只有当 terms 非空时才视为可用
#         __poly_available = len(__poly_terms) > 0

#     except Exception:
#         # 任意异常都回退到默认 Complexity
#         __poly_available = False
#         __poly_terms = []
#         __poly_intercept = 0.0
#         __poly_feature_names = None


# def __predict_with_poly(base_feats: Dict[str, float]) -> Optional[float]:
#     """使用已加载的多项式公式计算评分；若不可用返回 None。"""
#     __load_poly_model()
#     if not __poly_available or not __poly_terms or not __poly_feature_names:
#         return None

#     # 再次导入（防止上层环境热重载）
#     try:
#         from sklearn.preprocessing import PolynomialFeatures  # type: ignore
#     except Exception:
#         return None

#     # 1) 按训练时顺序取基础特征
#     xbase = np.array([[float(base_feats[k]) for k in _BASE_ORDER]], dtype=float)

#     # 2) 生成同顺序的多项式项
#     pf = PolynomialFeatures(degree=__poly_degree, include_bias=False)
#     pf.fit(np.zeros((1, len(_BASE_ORDER))))
#     Xp = pf.transform(xbase)
#     names = list(pf.get_feature_names_out(_BASE_ORDER))
#     name_to_idx = {n: i for i, n in enumerate(names)}

#     # 3) 组装系数向量
#     coef_vec = np.zeros(Xp.shape[1], dtype=float)
#     for t in __poly_terms:
#         n = t.get("name")
#         c = float(t.get("coef", 0.0))
#         idx = name_to_idx.get(n)
#         if idx is not None:
#             coef_vec[idx] = c

#     # 4) 线性组合 + 截距
#     y = float(np.dot(Xp[0], coef_vec) + __poly_intercept)
#     # 如果目标是概率/成功率，通常裁剪到 [0,1]
#     return float(np.clip(y, 0.0, 1.0))


# def predict_complexity(cfg: Dict[str, Any],
#                        connectivity: int = 4,
#                        K: int = 30,
#                        seed: int = 42,
#                        use_poly: bool = True) -> float:
#     """
#     传入一张地图配置 dict（至少包含 grid/num_agents；可选 starts/goals），返回复杂度分数。
#     优先使用你训练得到的“二次多项式 + 非负 LassoCV”公式（weights_poly.json）；
#     若不可用或没安装 sklearn，则回退到 complexity.py 的默认线性组合分数。
#     """
#     grid = np.array(cfg["grid"], dtype=np.uint8)
#     agents = int(cfg.get("num_agents", 2))
#     starts = cfg.get("starts", None)
#     goals = cfg.get("goals", None)

#     # 先用 complexity 计算基础特征 & 默认 Complexity
#     res = compute_complexity(
#         grid=grid,
#         starts=starts,
#         goals=goals,
#         agents=agents,
#         connectivity=connectivity,
#         K=K,
#         random_state=seed,
#         # 不在这里传线性 weights；默认内部会用自身的 DEFAULT_WEIGHTS
#     )

#     if use_poly:
#         base = {k: float(res[k]) for k in _BASE_ORDER}
#         y = __predict_with_poly(base)
#         if y is not None:
#             return y

#     # 回退
#     return float(res["Complexity"])

# def gen_random_grid(size: int,
#                     density: float,
#                     rng: np.random.Generator,
#                     border: bool = True) -> np.ndarray:
#     """
#     生成 size x size 的 0/1 网格。1=障碍，0=可走。
#     density: 障碍比例（0~1）；默认四周加一圈墙体，避免出界。
#     """
#     g = (rng.random((size, size)) < float(density)).astype(np.uint8)
#     if border:
#         g[0, :] = 1; g[-1, :] = 1; g[:, 0] = 1; g[:, -1] = 1
#     # 兜底：避免全障碍或全空（影响后续起终点采样）
#     if g.mean() < 0.02:
#         k = max(1, int(0.02 * size * size))
#         idx = rng.choice(size * size, size=k, replace=False)
#         g.flat[idx] = 1
#     if g.mean() > 0.98:
#         k = max(1, int(0.02 * size * size))
#         idx = rng.choice(size * size, size=k, replace=False)
#         g.flat[idx] = 0
#     return g

# def sample_starts_goals(grid: np.ndarray,
#                         num_agents: int,
#                         rng: np.random.Generator) -> Tuple[Optional[List[List[int]]],
#                                                            Optional[List[List[int]]]]:
#     """
#     在可行走单元随机采样起点/终点，各 num_agents 个。
#     返回：list[list[int,int]]；如果可用格不足则返回 (None, None)。
#     """
#     free = np.argwhere(grid == 0)
#     need = 2 * num_agents
#     if len(free) < max(2, need):
#         return None, None
#     idx = rng.choice(len(free), size=need, replace=False)
#     pts = [free[i].tolist() for i in idx]   # ← list 而不是 tuple，避免 !!python/tuple
#     starts = pts[:num_agents]
#     goals  = pts[num_agents:]
#     return starts, goals

# def generate_map_settings(map_sizes,
#                           agent_counts,
#                           densities,
#                           seed: int = 42,
#                           out_dir: str = "g2rl",
#                           out_name: str = "map_settings_generated.yaml"):
#     """
#     只生成 YAML（列表结构），每个元素包含：
#       - name / size / num_agents / density / obs_radius / max_episode_steps
#       - grid: 0/1 矩阵（list of list）
#       - starts, goals: list of [r, c] ；可能为 None
#     """
#     rng = np.random.default_rng(seed)
#     combinations = list(product(map_sizes, agent_counts, densities))
#     maps_yaml_list = []

#     os.makedirs(out_dir, exist_ok=True)

#     for i, (size, agents, density) in enumerate(combinations):
#         name = f"map_{i}"
#         grid = gen_random_grid(size=size, density=density, rng=rng, border=True)
#         starts, goals = sample_starts_goals(grid, agents, rng)

#         maps_yaml_list.append({
#             "name": name,
#             "size": int(size),
#             "num_agents": int(agents),
#             "density": float(density),
#             "obs_radius": 5,
#             "max_episode_steps": 100,
#             "grid": grid.astype(int).tolist(),
#             "starts": starts,     # list[list[int,int]] 或 None
#             "goals": goals        # list[list[int,int]] 或 None
#         })

#     yaml_path = os.path.join(out_dir, out_name)
#     with open(yaml_path, "w", encoding="utf-8") as f:
#         # 用 SafeDumper，避免非标准标签（如 !!python/tuple）
#         yaml.dump(maps_yaml_list, f, allow_unicode=True, Dumper=yaml.SafeDumper)
#     print(f"✅ 已生成 {len(maps_yaml_list)} 张地图，写入 {yaml_path}")
#     return yaml_path

# if __name__ == '__main__':
#     map_sizes = [32, 64, 96, 128]
#     agent_counts = [2, 4, 8, 16]
#     densities = [0.1, 0.3, 0.5, 0.7]

#     generate_map_settings(
#         map_sizes=map_sizes,
#         agent_counts=agent_counts,
#         densities=densities,
#         seed=42,
#         out_dir="g2rl",
#         out_name="map_settings_generated.yaml"
#     )


# generate_maps_with_grid_only.py
from itertools import product
import os
import yaml
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# 依赖你已有的 complexity.py（已加入 FRA、FDA）
from g2rl.complexity import compute_complexity

# 多项式权重文件（如果训练过，会放在 g2rl/ 目录）
_POLY_JSON_PATH = os.path.join(os.path.dirname(__file__), "weights_poly.json")

# 训练时的基础特征顺序（默认集，含 FRA/FDA）
_BASE_ORDER_DEFAULT = ["Size", "Agents", "Density", "LDD", "BN", "MC", "DLR", "FRA", "FDA"]

# 懒加载缓存
__poly_loaded = False
__poly_terms = None            # list[{"name": str, "coef": float}]
__poly_intercept = 0.0
__poly_degree = 2
__poly_feature_names_all = None    # PolynomialFeatures 展开的所有项名
__poly_base_feature_names = None   # 训练时的基础特征顺序（输入顺序）
__poly_available = False           # 是否可用（文件存在且依赖可用）

def _to_builtin(obj):
    """把嵌套结构深度转换成可被 SafeDumper 序列化的纯 Python 类型。"""
    import numpy as _np
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    if isinstance(obj, (_np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, tuple):   # 保险：任何 tuple 都转 list
        return [_to_builtin(x) for x in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    return str(obj)

def __load_poly_model():
    """尝试读取 weights_poly.json 并准备特征名；失败则保持回退模式。"""
    global __poly_loaded, __poly_terms, __poly_intercept, __poly_degree
    global __poly_feature_names_all, __poly_base_feature_names, __poly_available

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

        # 优先使用文件里保存的训练时基础特征顺序；没有就用默认
        base_from_file = data.get("base_features", None)
        __poly_base_feature_names = list(base_from_file) if base_from_file else list(_BASE_ORDER_DEFAULT)

        # 仅用于生成稳定的项名称顺序（不需要真实数据）
        from sklearn.preprocessing import PolynomialFeatures  # type: ignore
        pf = PolynomialFeatures(degree=__poly_degree, include_bias=False)
        pf.fit(np.zeros((1, len(__poly_base_feature_names))))
        __poly_feature_names_all = list(pf.get_feature_names_out(__poly_base_feature_names))

        # 只有当 terms 非空时才视为可用
        __poly_available = len(__poly_terms) > 0

    except Exception:
        # 任意异常都回退到默认 Complexity
        __poly_available = False
        __poly_terms = []
        __poly_intercept = 0.0
        __poly_feature_names_all = None
        __poly_base_feature_names = None


def __predict_with_poly(base_feats: Dict[str, float]) -> Optional[float]:
    """使用已加载的多项式公式计算评分；若不可用返回 None。"""
    __load_poly_model()
    if not __poly_available or not __poly_terms or not __poly_feature_names_all or not __poly_base_feature_names:
        return None

    # 再次导入（防止上层环境热重载）
    try:
        from sklearn.preprocessing import PolynomialFeatures  # type: ignore
    except Exception:
        return None

    # 1) 按“训练时基础特征顺序”组织输入；缺失的特征用 0 补齐
    x_vec = [float(base_feats.get(k, 0.0)) for k in __poly_base_feature_names]
    xbase = np.array([x_vec], dtype=float)

    # 2) 生成同 degree 的多项式项
    pf = PolynomialFeatures(degree=__poly_degree, include_bias=False)
    pf.fit(np.zeros((1, len(__poly_base_feature_names))))
    Xp = pf.transform(xbase)
    names = list(pf.get_feature_names_out(__poly_base_feature_names))
    name_to_idx = {n: i for i, n in enumerate(names)}

    # 3) 组装系数向量（按名字对齐）
    coef_vec = np.zeros(Xp.shape[1], dtype=float)
    for t in __poly_terms:
        n = t.get("name")
        c = float(t.get("coef", 0.0))
        idx = name_to_idx.get(n)
        if idx is not None:
            coef_vec[idx] = c

    # 4) 线性组合 + 截距；若目标是概率/成功率，裁剪到 [0,1]
    y = float(np.dot(Xp[0], coef_vec) + __poly_intercept)
    return float(np.clip(y, 0.0, 1.0))


def predict_complexity(cfg: Dict[str, Any],
                       connectivity: int = 4,
                       K: int = 30,
                       seed: int = 42,
                       use_poly: bool = True) -> Tuple[float, Dict[str, float]]:
    grid = np.array(cfg["grid"], dtype=np.uint8)
    agents = int(cfg.get("num_agents", 2))
    starts = cfg.get("starts", None)
    goals = cfg.get("goals", None)

    # ✅ 仅传入 compute_complexity 时转为 tuple；cfg 本身保持 list 以便 YAML
    starts_t = [tuple(map(int, s)) for s in starts] if starts else None
    goals_t  = [tuple(map(int, g)) for g in goals] if goals else None

    res = compute_complexity(
        grid=grid,
        starts=starts_t,
        goals=goals_t,
        agents=agents,
        connectivity=connectivity,
        K=K,
        random_state=seed,
    )

    # 组装用于多项式的基础特征（和训练时的 base_features 对齐）
    base_feats = {k: float(res.get(k, 0.0)) for k in _BASE_ORDER_DEFAULT}

    score_poly = None
    if use_poly:
        score_poly = __predict_with_poly(base_feats)

    score = float(score_poly) if score_poly is not None else float(res["Complexity"])

    # 精简的指标字典（落盘）
    metrics = {
        "Size": float(res["Size"]),
        "Agents": float(res["Agents"]),
        "Density": float(res["Density"]),
        "LDD": float(res["LDD"]),
        "BN": float(res["BN"]),
        "MC": float(res["MC"]),
        "DLR": float(res["DLR"]),
        "FRA": float(res.get("FRA", 0.0)),
        "FDA": float(res.get("FDA", 0.0)),
        "FRA_hard": float(res.get("FRA_hard", 0.0)),
        "FDA_ratio": float(res.get("FDA_ratio", 1.0)),
        "Complexity_linear": float(res["Complexity"]),   # 线性回退版
    }
    return score, metrics


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
    pts = [tuple(free[i]) for i in idx]     # 返回 tuple (r, c)
   # list 而不是 tuple，避免 !!python/tuple
    starts = pts[:num_agents]
    goals  = pts[num_agents:]
    return starts, goals


def generate_map_settings(map_sizes,
                          agent_counts,
                          densities,
                          seed: int = 42,
                          out_dir: str = "g2rl",
                          out_name: str = "map_settings_generated.yaml",
                          with_metrics: bool = True,
                          use_poly: bool = True):
    """
    生成 YAML（列表结构），每个元素包含：
      - name / size / num_agents / density / obs_radius / max_episode_steps
      - grid: 0/1 矩阵（list of list）
      - starts, goals: list of [r, c] ；可能为 None
      - 可选：complexity_score（多项式或线性回退） 和 metrics（含 FRA/FDA）
    """
    rng = np.random.default_rng(seed)
    combinations = list(product(map_sizes, agent_counts, densities))
    maps_yaml_list = []

    os.makedirs(out_dir, exist_ok=True)

    for i, (size, agents, density) in enumerate(combinations):
        name = f"map_{i}"
        grid = gen_random_grid(size=size, density=density, rng=rng, border=True)
        starts, goals = sample_starts_goals(grid, agents, rng)

        item = {
            "name": name,
            "size": int(size),
            "num_agents": int(agents),
            "density": float(density),
            "obs_radius": 5,
            "max_episode_steps": 100,
            "grid": grid.astype(int).tolist(),
            "starts": starts,     # list[list[int,int]] 或 None
            "goals": goals        # list[list[int,int]] 或 None
        }

        if with_metrics:
            score, metrics = predict_complexity(
                cfg=item, connectivity=4, K=30, seed=seed, use_poly=use_poly
            )
            item["complexity_score"] = float(score)
            item["metrics"] = metrics  # 包含 FRA/FDA/LDD/BN/MC/DLR 等

        maps_yaml_list.append(item)

    yaml_path = os.path.join(out_dir, out_name)
    safe_yaml_data = _to_builtin(maps_yaml_list) 
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(safe_yaml_data, f, allow_unicode=True, Dumper=yaml.SafeDumper)
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
        out_name="map_settings_generated_new.yaml",
        with_metrics=True,   # ← 建议开启，落盘 FRA/FDA 等
        use_poly=True       # ← 若无 weights_poly.json 或未装 sklearn 会自动回退
    )
