# -*- coding: utf-8 -*-
"""
infer_complexity.py — 固定配置版（无需命令行参数）
功能：
- 加载训练好的 NN 模型（nn_model.pt）
- 读取地图特征 CSV
- 预测 success_rate，并计算 complexity = 1 - success_rate（可裁剪到[0,1]）
- 写回到新的 CSV

直接运行： python infer_complexity.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============ 你的固定配置 ============
MODEL_PATH  = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main-nn/0827/nn_model.pt"
INPUT_CSV   = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main-nn/train_gray3d-Copy-FDA.csv"
OUTPUT_CSV  = r"C:/Users/MSc_SEIoT_1/MAPF_G2RL-main-nn/0827result/maps_features_with_complexity.csv"
CLIP        = True          # 是否裁剪到 [0,1]
ID_COL      = None          # 如有 map 名列（如 'map_name'），填列名；否则 None
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# =====================================

# 与训练时一致的 MLP 结构
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, hidden_layers: int, dropout: float, use_sigmoid_head: bool):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(hidden_layers):
            layers += [
                nn.Linear(dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.backbone = nn.Sequential(*layers)
        self.use_sigmoid = use_sigmoid_head

    def forward(self, x):
        out = self.backbone(x).squeeze(-1)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out

def safe_standardize(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    stds_safe = np.where(stds == 0, 1.0, stds)
    return (X - means) / stds_safe

def main():
    # 1) 加载 checkpoint
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到模型文件：{MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location="cpu")

    in_dim       = ckpt["in_dim"]
    hidden_dim   = ckpt["hidden_dim"]
    hidden_layers= ckpt["hidden_layers"]
    dropout      = ckpt["dropout"]
    use_sigmoid  = ckpt.get("use_sigmoid_head", True)

    feature_columns = ckpt.get("feature_columns", ckpt.get("features_used"))
    if feature_columns is None:
        raise ValueError("模型中缺少 feature_columns / features_used，无法确定特征列顺序。")

    scaler_means = np.array(ckpt["scaler_means"], dtype=float)
    scaler_stds  = np.array(ckpt["scaler_stds"], dtype=float)
    if len(feature_columns) != in_dim or scaler_means.shape[0] != in_dim:
        raise ValueError(f"in_dim({in_dim}) 与特征/标准化器维度不一致。")

    # 2) 设备与模型
    device = torch.device(DEVICE)
    print(f"🖥 设备: {device}")

    model = MLP(in_dim=in_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                dropout=dropout,
                use_sigmoid_head=use_sigmoid).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 3) 读取输入 CSV
    # if not os.path.exists(INPUT_CSV):
    #     raise FileNotFoundError(f"未找到输入 CSV：{INPUT_CSV}")
    # df = pd.read_csv(INPUT_CSV)
    # print(f"📄 输入CSV: {INPUT_CSV}  |  形状: {df.shape}")

    # missing = [c for c in feature_columns if c not in df.columns]
    # if missing:
    #     raise ValueError(f"输入CSV缺少以下特征列（需与训练一致）: {missing}")

    # # 4) 选择可推理的行（特征非 NaN）
    # mask_ok = ~df[feature_columns].isna().any(axis=1)
    # num_ok = int(mask_ok.sum())
    # print(f"✅ 可推理样本数: {num_ok} / {len(df)}")

    # X  = df.loc[mask_ok, feature_columns].astype(float).values
    # Xs = safe_standardize(X, scaler_means, scaler_stds)
    # 3) 读取输入 CSV
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"未找到输入 CSV：{INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"📄 输入CSV: {INPUT_CSV}  |  形状: {df.shape}")

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"输入CSV缺少以下特征列（需与训练一致）: {missing}")

# === 3.1 清洗：把特征列里的非数字字符清掉，再转为 float ===
    import re
    def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    # 保留数字、正负号、小数点、科学计数法 e/E；其余都去掉
        cleaned = s.astype(str).str.replace(r"[^0-9eE\+\-\.]", "", regex=True)
    # 将空字符串替换为 NaN
        cleaned = cleaned.replace({"": np.nan})
    # 安全转换为数值，非法的记为 NaN
        return pd.to_numeric(cleaned, errors="coerce")

    before_na = df[feature_columns].isna().sum()
    for col in feature_columns:
        df[col] = _coerce_numeric_series(df[col])

    after_na = df[feature_columns].isna().sum()
    added_na = (after_na - before_na)
    added_na = added_na[added_na > 0]
    if not added_na.empty:
        print("⚠️ 清洗后转为 NaN 的条目（说明原本含有非数字字符）:")
        for c, n in added_na.items():
            print(f"   - {c}: +{int(n)}")

# 4) 选择可推理的行（特征非 NaN）
    mask_ok = ~df[feature_columns].isna().any(axis=1)
    num_ok = int(mask_ok.sum())
    num_all = len(df)
    print(f"✅ 可推理样本数: {num_ok} / {num_all}")
    if num_ok == 0:
    # 打印几行示例，帮助你定位问题数据
        print("❌ 清洗后没有可用样本。示例前5行：")
        print(df[feature_columns].head())
        raise ValueError("所有特征行存在 NaN，无法推理。请检查原始数据格式。")

# 5) 标准化并推理
    X  = df.loc[mask_ok, feature_columns].values.astype(float)
    Xs = safe_standardize(X, scaler_means, scaler_stds)


    # 5) 推理
    with torch.no_grad():
        x_tensor = torch.tensor(Xs, dtype=torch.float32, device=device)
        pred_sr = model(x_tensor).cpu().numpy().reshape(-1)

    if CLIP or use_sigmoid:
        pred_sr = np.clip(pred_sr, 0.0, 1.0)

    complexity = 1.0 - pred_sr
    if CLIP or use_sigmoid:
        complexity = np.clip(complexity, 0.0, 1.0)

    # 6) 回填 & 保存
    pred_col = "nn_pred_success_rate"
    comp_col = "nn_complexity"

    df[pred_col] = np.nan
    df.loc[mask_ok, pred_col] = pred_sr
    df[comp_col] = np.nan
    df.loc[mask_ok, comp_col] = complexity

    out_dir = os.path.dirname(OUTPUT_CSV)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"💾 已保存: {OUTPUT_CSV}")

    # 7) 可选：打印 Top-5（仅当提供 ID_COL 且存在时）
    if ID_COL and ID_COL in df.columns:
        sub = df.loc[mask_ok, [ID_COL, comp_col]].dropna()
        if not sub.empty:
            worst = sub.sort_values(comp_col, ascending=False).head(5)
            best  = sub.sort_values(comp_col, ascending=True).head(5)
            print("\n🏴‍☠️ Top-5 最复杂：")
            for _, r in worst.iterrows():
                print(f"  {r[ID_COL]}  -> complexity={r[comp_col]:.4f}")
            print("\n🍃 Top-5 最简单：")
            for _, r in best.iterrows():
                print(f"  {r[ID_COL]}  -> complexity={r[comp_col]:.4f}")
    else:
        print("ℹ️ 未配置 ID_COL 或输入CSV中不存在该列，跳过 Top-5 打印。")

    print("✅ 推理完成。")

if __name__ == "__main__":
    main()
