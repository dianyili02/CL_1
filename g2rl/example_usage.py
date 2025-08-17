
import numpy as np
from complexity_module import compute_map_complexity
# #随机生成地图后计算
# INTERCEPT = 0.848 
# WEIGHTS = {
#     'Size': 0.01,
#     'Agents': 0.1,
#     'Density': 0.8,
# }

# H, W = 32, 32
# density = 0.2
# rng = np.random.default_rng(0)
# grid = (rng.random((H, W)) < density).astype(np.uint8)
# grid_list = grid.tolist()
# complexity, used, raw = compute_map_complexity(
#     {"grid": grid_list, "num_agents": 4},
#     intercept=INTERCEPT,
#     weights=WEIGHTS,
#     size_mode="max",
# )


# print('Complexity:', complexity)
# print('Used features:', used)
# print('Raw features:', raw)

# #对单个yaml地图文件
# from complexity_module import compute_map_complexity

# INTERCEPT = ...
# WEIGHTS = {...}

# cpx, used, raw = compute_map_complexity(
#     "/path/to/map.yaml",
#     intercept=INTERCEPT,
#     weights=WEIGHTS,
#     # feature_mean_std={"Size": (mean, std), ...}  # 如果训练时做过标准化，传入一致的均值/方差
#     size_mode="max",  # 或 'area' / 'diag'，务必与训练口径一致
# )


#批量处理一个目录（可选写回 YAML）
import pandas as pd

# === 你的公式 ===
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

def compute_complexity_from_row(row):
    z = INTERCEPT
    for feat, w in WEIGHTS.items():
        if feat in row and pd.notna(row[feat]):
            z += w * row[feat]
    return z

# === 读取你的地图 CSV ===
df = pd.read_csv("C:/Users/MSc_SEIoT_1/MAPF_G2RL-main/logs/train_gray3d-Copy.csv")

# === 计算 Complexity 列 ===
df["Complexity"] = df.apply(compute_complexity_from_row, axis=1)

# === 保存新的 CSV ===
out_path = "map_complexity.csv"
df.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"✅ 已生成 {out_path}")

