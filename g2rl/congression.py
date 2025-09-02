# -*- coding: utf-8 -*-
"""
非线性建模（神经网络 MLP）— 适用于 MAPF/CL 成功率或复杂度回归
功能：
1) 读CSV → 自动找目标列 → 别名映射特征 → 清洗 → 标准化
2) MLP 训练（含Dropout + L2正则 + 早停） + 验证指标（R/R2/MAE/RMSE/MAPE）
3) 写回预测/残差列到CSV；保存诊断图；保存模型与标准化器
"""

import os, json, math, copy, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ============ 配置（按需改） ============
SAVE_DIR = r"C:/Users/dell/Desktop/CL_1/0902"  # ← 你想保存的目录
os.makedirs(SAVE_DIR, exist_ok=True)  # 若不存在则创建

CSV_CANDIDATES = [
    r"C:/Users/dell/Desktop/CL_1/train_gray3d-Copy-FDA.csv",
]
TARGET_COL = None  # None = 自动识别（success_rate / Complexity / complexity / target / y）

CANDIDATE_FEATURES = [
    "size", "num_agents", "density", "density_actual",
    "LDD", "BN", "MC", "DLR", "FRA", "FDA"
]

# 训练相关
RANDOM_STATE      = 42
TEST_SIZE         = 0.2
BATCH_SIZE        = 128
MAX_EPOCHS        = 500
PATIENCE          = 30           # 早停容忍轮数（按 val RMSE）
LEARNING_RATE     = 1e-3
WEIGHT_DECAY      = 1e-4         # L2 正则
HIDDEN_DIM        = 128
HIDDEN_LAYERS     = 2            # 隐藏层层数（不含输入/输出层）
DROPOUT           = 0.10
USE_SIGMOID_HEAD  = True         # 如果目标是[0,1]（成功率），建议 True

# 输出命名
OUTPUT_SUFFIX   = "_nn_with_pred.csv"
FIG_PRED_ACT    = "_nn_pred_vs_actual.png"
FIG_RESID_HIST  = "_nn_residual_hist.png"
FIG_RESID_FIT   = "_nn_residuals_vs_fitted.png"
MODEL_PATH      = "_nn_model.pt"
SCALER_PATH     = "_nn_scaler.json"
# ======================================


def _find_csv_path(cands: List[str]) -> str:
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"找不到 CSV。请把 CSV 放在这些路径之一：\n{cands}")


def _pick_target_col(df: pd.DataFrame, pref=None) -> str:
    if pref and pref in df.columns:
        return pref
    for cand in ["success_rate", "Complexity", "complexity", "target", "y"]:
        if cand in df.columns:
            return cand
    raise ValueError("未找到目标列（success_rate/Complexity/complexity/target/y）。请设置 TARGET_COL。")


def _build_feature_mapping(df_cols):
    alias = {
        "size":           ["size", "Size"],
        "num_agents":     ["num_agents", "Agents", "agents", "n_agents"],
        "density":        ["density", "Density"],
        "density_actual": ["density_actual", "Density_actual", "actual_density"],
        "LDD":            ["LDD", "ldd"],
        "BN":             ["BN", "bn", "Bottleneck", "bottleneck"],
        "MC":             ["MC", "mc", "MapConnectivity", "map_connectivity"],  # 避免误映射到 moving_cost
        "DLR":            ["DLR", "dlr"],
        "FRA":            ["FRA", "fra"],
        "FDA":            ["FDA", "fda"],
        # 如你有 moving_cost，请另起规范名，避免和 MC 混淆
        "moving_cost":    ["MovingCost", "moving_cost", "MCost"],
    }
    lower_cols = {c.lower(): c for c in df_cols}
    mapping = {}
    for canonical, cands in alias.items():
        for c in cands:
            if c in df_cols:
                mapping[canonical] = c
                break
            if c.lower() in lower_cols:
                mapping[canonical] = lower_cols[c.lower()]
                break
    return mapping


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


def metrics_block(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # clip到[0,1]（若是成功率类目标）
    yt = y_true.astype(float)
    yp = y_pred.astype(float)
    if USE_SIGMOID_HEAD:
        yp = np.clip(yp, 0.0, 1.0)

    R = float(np.corrcoef(yt, yp)[0, 1]) if (len(yt) >= 2 and np.std(yt) > 0 and np.std(yp) > 0) else float("nan")
    R2 = float(r2_score(yt, yp))
    MAE = float(mean_absolute_error(yt, yp))
    RMSE = float(np.sqrt(mean_squared_error(yt, yp)))
    denom = np.where(np.abs(yt) < 1e-12, 1e-12, np.abs(yt))
    MAPE = float(np.mean(np.abs((yt - yp) / denom)) * 100.0)
    return {"R": R, "R2": R2, "MAE": MAE, "RMSE": RMSE, "MAPE%": MAPE}


def main():
    # 1) 读CSV
    csv_path = _find_csv_path(CSV_CANDIDATES)
    base, _ = os.path.splitext(csv_path)
    print("📄 CSV:", csv_path)
    df = pd.read_csv(csv_path)
    print("Columns:", list(df.columns), "Shape:", df.shape)

    # 2) 目标列
    target_col = _pick_target_col(df, TARGET_COL)
    print("🎯 目标列:", target_col)

    # 3) 特征映射（别名）
    mapping = _build_feature_mapping(df.columns)
    used_features = [f for f in CANDIDATE_FEATURES if f in mapping]
    if not used_features:
        raise ValueError("❌ 没找到任何可用特征，请检查列名或 CANDIDATE_FEATURES")

    print("🧩 使用特征（规范名）:", used_features)
    print("↳ 列映射:", {k: mapping[k] for k in used_features})

    # 4) 丢 NaN
    cols_need = [mapping[f] for f in used_features] + [target_col]
    dff = df[cols_need].dropna().copy()
    if dff.empty:
        raise ValueError("❌ 有效样本为 0。请先清洗数据。")
    print("✅ 可用样本数:", len(dff))

    # 5) 取X/y
    # X_all = dff[[mapping[f] for f in used_features]].astype(float).values


    cols = [mapping[f.strip()] for f in used_features]  # 注意 f.strip() 去掉 ' FDA' 前的空格问题

# 针对每一列做清洗
    clean = dff[cols].apply(
        lambda col: col.astype(str)  # 转成字符串
                   .str.replace(',', '', regex=False)
                   .str.replace('%', '', regex=False)
                   .str.replace(r'[^0-9eE+\-\.]', '', regex=True)  # 保留数值相关字符
    )

# 转换成数值，非法的转 NaN
    clean = clean.apply(pd.to_numeric, errors='coerce')

# 检查坏行
    bad_rows = clean.isna().any(axis=1)
    if bad_rows.any():
        print(f"[清洗] 有 {bad_rows.sum()} 行包含非数值特征：")
        print(dff.loc[bad_rows, cols].head())

# 丢弃坏行
    clean = clean.dropna(axis=0, how='any')

    X_all = clean.values


    y_all = dff[target_col].astype(float).values

    # 6) 划分训练/验证
    X_tr, X_va, y_tr, y_va = train_test_split(X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # 7) 标准化（仅 X）
    x_scaler = StandardScaler().fit(X_tr)
    X_tr_s = x_scaler.transform(X_tr)
    X_va_s = x_scaler.transform(X_va)

    # 8) 数据集与加载器
    ds_tr = TabularDataset(X_tr_s, y_tr)
    ds_va = TabularDataset(X_va_s, y_va)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 9) 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🖥 设备:", device)

    # 10) 模型、优化器、损失
    model = MLP(in_dim=X_tr_s.shape[1],
                hidden_dim=HIDDEN_DIM,
                hidden_layers=HIDDEN_LAYERS,
                dropout=DROPOUT,
                use_sigmoid_head=USE_SIGMOID_HEAD).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()  # 回归

    # 11) 训练（早停基于 val RMSE）
    best_state = None
    best_val_rmse = float("inf")
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(ds_tr)

        # 验证
        model.eval()
        with torch.no_grad():
            preds = []
            for xb, yb in dl_va:
                xb = xb.to(device)
                preds.append(model(xb).cpu().numpy())
            y_pred_va = np.concatenate(preds, axis=0)
        val_rmse = float(np.sqrt(mean_squared_error(y_va, y_pred_va)))

        if val_rmse + 1e-9 < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch:4d}] train_loss={train_loss:.6f}  val_RMSE={val_rmse:.6f}  best={best_val_rmse:.6f}")
        if no_improve >= PATIENCE:
            print(f"⏹️ 早停：{epoch} / {MAX_EPOCHS}，最佳 val_RMSE={best_val_rmse:.6f}")
            break

    # 加载最佳参数
    if best_state is not None:
        model.load_state_dict(best_state)

    # 12) 评估（train/val）
    model.eval()
    with torch.no_grad():
        y_pred_tr = model(torch.tensor(X_tr_s, dtype=torch.float32, device=device)).cpu().numpy()
        y_pred_va = model(torch.tensor(X_va_s, dtype=torch.float32, device=device)).cpu().numpy()

    m_tr = metrics_block(y_tr, y_pred_tr)
    m_va = metrics_block(y_va, y_pred_va)

    print("\n==== Train Metrics ====")
    for k, v in m_tr.items():
        print(f"{k:>7s}: {v: .6f}")
    print("==== Valid Metrics ====")
    for k, v in m_va.items():
        print(f"{k:>7s}: {v: .6f}")

    # 13) 全表预测（对原 df；缺失保留 NaN）
    #    仅对原 df 中“特征完整”的行做预测
    mask_ok = ~df[[mapping[f] for f in used_features]].isna().any(axis=1)
    # X_full = df.loc[mask_ok, [mapping[f] for f in used_features]].astype(float).values
    # ==== 统一修正特征名空格等问题 ====
    used_features = [f.strip() for f in used_features]
    cols = [mapping[f] for f in used_features]

# ==== 先对子表做数值清洗（逐列 .str 操作）====
    raw_feat = df.loc[:, cols].copy()

    def _clean_series(s):
        return (s.astype(str)
                .str.replace(',', '', regex=False)      # 去掉千分位
                .str.replace('%', '', regex=False)      # 去掉百分号
                .str.replace(r'[^0-9eE+\.-]', '', regex=True))  # 仅保留数字/正负号/小数点/eE

    clean_feat = raw_feat.apply(_clean_series)
    num_feat = clean_feat.apply(pd.to_numeric, errors='coerce')

# ==== 记录并处理坏行 ====
    bad_rows = num_feat.isna().any(axis=1)
    if bad_rows.any():
        print(f"[清洗] 发现 {bad_rows.sum()} 行含非数值特征，将被剔除。示例：")
    # 打印前 5 行问题样本，便于定位哪一列脏数据
        print(raw_feat[bad_rows].head(5))

# 如果你已有 mask_ok（比如目标列等规则），把两种掩码都用上
    mask_good = (~bad_rows)
    if 'mask_ok' in locals():
        mask_good = mask_good & mask_ok

# ==== 最终特征矩阵 ====
    X_full = num_feat.loc[mask_good, :].values

# ==== 若已有 y，也需同步掩码（示例）====
# y_full = y_series.loc[mask_good].values

    X_full_s = x_scaler.transform(X_full)
    with torch.no_grad():
        pred_full = model(torch.tensor(X_full_s, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)

    pred_col  = f"nn_pred_{target_col}"
    resid_col = f"nn_residual_{target_col}"
    df[pred_col] = np.nan
    df.loc[mask_ok, pred_col] = pred_full
    # 仅对目标非缺失计算残差
    mask_resid = mask_ok & df[target_col].notna()
    df[resid_col] = np.nan
    df.loc[mask_resid, resid_col] = df.loc[mask_resid, target_col].astype(float).values - df.loc[mask_resid, pred_col].astype(float).values

    # 14) 保存新CSV
    out_csv = os.path.join(SAVE_DIR, "nn_results_with_pred.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 已写出：{out_csv}")

    # 15) 诊断图（基于验证集）
    #   图1：Pred vs Actual
    fig1       = os.path.join(SAVE_DIR, "nn_pred_vs_actual.png")
    plt.figure()
    plt.scatter(y_va, y_pred_va, alpha=0.7)
    minv = float(min(np.min(y_va), np.min(y_pred_va)))
    maxv = float(max(np.max(y_va), np.max(y_pred_va)))
    plt.plot([minv, maxv], [minv, maxv], linestyle="--")
    plt.xlabel("Actual (Valid)")
    plt.ylabel("Predicted (Valid)")
    plt.title(f"NN Pred vs Actual (Valid)\nR={m_va['R']:.3f}, R²={m_va['R2']:.3f}, MAE={m_va['MAE']:.3f}, RMSE={m_va['RMSE']:.3f}")
    plt.tight_layout()
    plt.savefig(fig1, dpi=160); plt.close()
    print(f"🖼 已保存图：{fig1}")

    #   图2：Residual Histogram
    fig2       = os.path.join(SAVE_DIR, "nn_residual_hist.png")
    plt.figure()
    plt.hist(y_va - y_pred_va, bins=30)
    plt.xlabel("Residual (y - y_pred) [Valid]")
    plt.ylabel("Count")
    plt.title("NN Residuals Histogram (Valid)")
    plt.tight_layout()
    plt.savefig(fig2, dpi=160); plt.close()
    print(f"🖼 已保存图：{fig2}")

    #   图3：Residuals vs Fitted
    fig3       = os.path.join(SAVE_DIR, "nn_residuals_vs_fitted.png")
    plt.figure()
    plt.scatter(y_pred_va, y_va - y_pred_va, alpha=0.7)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Fitted (Predicted) [Valid]")
    plt.ylabel("Residual [Valid]")
    plt.title("NN Residuals vs Fitted (Valid)")
    plt.tight_layout()
    plt.savefig(fig3, dpi=160); plt.close()
    print(f"🖼 已保存图：{fig3}")

    # 16) 保存模型与标准化器
    model_path = os.path.join(SAVE_DIR, "nn_model.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": X_tr_s.shape[1],
        "hidden_dim": HIDDEN_DIM,
        "hidden_layers": HIDDEN_LAYERS,
        "dropout": DROPOUT,
        "use_sigmoid_head": USE_SIGMOID_HEAD,
        "scaler_means": x_scaler.mean_.tolist(),
        "scaler_stds": x_scaler.scale_.tolist(),
        "features_used": used_features,
        "feature_columns": [mapping[f] for f in used_features],
        "target": target_col,
    }, model_path)
    print(f"🧠 模型已保存：{model_path}")

    scaler_meta = {
        "means": x_scaler.mean_.tolist(),
        "stds": x_scaler.scale_.tolist(),
        "feature_order": [mapping[f] for f in used_features],
        "standardize": True
    }
    scaler_path= os.path.join(SAVE_DIR, "nn_scaler.json")
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler_meta, f, ensure_ascii=False, indent=2)
    print(f"📝 标准化器信息已保存：{scaler_path}")

if __name__ == "__main__":
    main()
