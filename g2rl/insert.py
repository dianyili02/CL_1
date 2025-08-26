import pandas as pd
import yaml
import numpy as np
import json
import os


# 文件路径
YAML_PATH = "C:/Users/MSc_SEIoT_1/MAPF_G2RL-main - train/g2rl/map_settings_generated_new.yaml"
CSV_IN = "C:/Users/MSc_SEIoT_1/MAPF_G2RL-main/logs/train_gray3d-Copy.csv"
CSV_OUT = "C:/Users/MSc_SEIoT_1/MAPF_G2RL-main - train/train_gray3d-Copy.csv"


# ---------- 工具 ----------
def to_grid(arr):
    g = np.array(arr, dtype=np.uint8)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")
    return (g > 0).astype(np.uint8)

def compute_FRA(grid: np.ndarray) -> float:
    H, W = grid.shape
    free = np.argwhere(grid == 0)
    if free.size == 0:
        return 0.0
    total = 0.0
    for x, y in free:
        neigh = free_neigh = 0
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < H and 0 <= ny < W:
                neigh += 1
                if grid[nx, ny] == 0:
                    free_neigh += 1
        if neigh > 0:
            total += free_neigh / neigh
    return float(total / len(free))

def compute_FDA_from_grid(grid: np.ndarray) -> float:
    return float((grid == 0).sum() / grid.size)

def pick_name_column(df: pd.DataFrame):
    candidates = ["map_name", "name", "map", "map_id", "Map", "Name"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def parse_config_json_column(df: pd.DataFrame):
    """若 CSV 有 config_json 列，将其中的 size/num_agents/density/obs_radius/max_episode_steps 解析出成列。"""
    if "config_json" not in df.columns:
        return df
    # 避免破坏原 DataFrame
    df = df.copy()
    ks = ["size","num_agents","density","obs_radius","max_episode_steps"]
    # 若已存在这些列则不覆盖
    need = [k for k in ks if k not in df.columns]
    if not need:
        return df
    parsed = {k: [] for k in need}
    for s in df["config_json"].astype(str).fillna("{}"):
        try:
            obj = json.loads(s)
        except Exception:
            obj = {}
        for k in need:
            parsed[k].append(obj.get(k, np.nan))
    for k in need:
        df[k] = parsed[k]
    return df

def round_float_cols(df: pd.DataFrame, cols, ndigits=6):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(ndigits)
    return df

# ---------- 读取 YAML ----------
with open(YAML_PATH, "r", encoding="utf-8") as f:
    ydata = yaml.safe_load(f)

# 顶层 list -> dict
if isinstance(ydata, list):
    ydict = {(m.get("name") or f"map_{i}"): m for i, m in enumerate(ydata)}
elif isinstance(ydata, dict):
    ydict = ydata
else:
    raise ValueError("Unsupported YAML structure.")

# ---------- 从 YAML 生成 FRA/FDA + 配置键 的表 ----------
rows = []
for key, spec in ydict.items():
    map_name = spec.get("name", key)

    # 配置字段（可能用于合并）
    size  = spec.get("size", np.nan)
    nag   = spec.get("num_agents", np.nan)
    dens  = spec.get("density", spec.get("Density", np.nan))
    obsr  = spec.get("obs_radius", np.nan)
    msteps= spec.get("max_episode_steps", np.nan)

    fra = spec.get("FRA", None)
    fda = spec.get("FDA", None)

    grid = None
    if fra is None or fda is None:
        if "grid" in spec and spec["grid"] is not None:
            try:
                grid = to_grid(spec["grid"])
            except Exception:
                grid = None
        if fra is None:
            fra = compute_FRA(grid) if grid is not None else np.nan
        if fda is None:
            if grid is not None:
                fda = compute_FDA_from_grid(grid)
            else:
                try:
                    dens_f = float(dens) if dens is not None else np.nan
                except Exception:
                    dens_f = np.nan
                fda = (1.0 - dens_f) if pd.notna(dens_f) else np.nan

    rows.append({
        "map_name": map_name,
        "size": size,
        "num_agents": nag,
        "density": dens,
        "obs_radius": obsr,
        "max_episode_steps": msteps,
        "FRA": fra,
        "FDA": fda,
    })

ydf = pd.DataFrame(rows)

# 统一数值精度避免浮点误差
num_keys = ["size","num_agents","density","obs_radius","max_episode_steps"]
ydf = round_float_cols(ydf, num_keys, ndigits=6)

# ---------- 读取 CSV ----------
df = pd.read_csv(CSV_IN)
df = parse_config_json_column(df)  # 从 config_json 解析配置键（若有）
df = round_float_cols(df, num_keys, ndigits=6)

# ---------- 选择合并策略 ----------
name_col = pick_name_column(df)

merged = None
used_key = None

if name_col is not None and "map_name" in ydf.columns:
    # 先用名字合并
    merged = df.merge(ydf[["map_name","FRA","FDA"]].rename(columns={"map_name":name_col}),
                      on=name_col, how="left")
    used_key = f"name_col:{name_col}"

if merged is None:
    # 尝试用配置键合并（五列全在才用）
    keyset = [k for k in num_keys if k in df.columns and k in ydf.columns]
    if set(["size","num_agents","density","obs_radius","max_episode_steps"]).issubset(set(keyset)):
        merged = df.merge(ydf[keyset + ["FRA","FDA"]], on=keyset, how="left")
        used_key = "config_keys: size,num_agents,density,obs_radius,max_episode_steps"

if merged is None and "config_json" in df.columns:
    # 兜底：如果 parse 失败或列缺失，也尽量用能对齐的子集
    keyset = [k for k in num_keys if k in df.columns and k in ydf.columns]
    if keyset:
        merged = df.merge(ydf[keyset + ["FRA","FDA"]], on=keyset, how="left")
        used_key = f"config_keys_subset: {keyset}"

if merged is None:
    raise KeyError(
        "无法找到可用于合并的键！\n"
        f"CSV 列：{list(df.columns)}\n"
        f"YAML 可用键：{num_keys} + map_name\n"
        "建议：确保 CSV 至少包含 size/num_agents/density/obs_radius/max_episode_steps 这五列 "
        "或提供地图名列（map_name/name/map_id）。"
    )

# 统计匹配情况
miss = merged["FRA"].isna().sum()
print(f"🔗 Merge key used: {used_key}")
print(f"✅ 合并完成：总行数 {len(merged)}，其中 FRA/FDA 缺失行数 {miss}")

# ---------- 保存 ----------
os.makedirs(os.path.dirname(CSV_OUT) or ".", exist_ok=True)
merged.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
print(f"💾 已写出：{CSV_OUT}")
