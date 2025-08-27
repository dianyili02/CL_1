# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Constrained linear regression with sign constraints (cvxpy).
# - 自动读取 CSV（默认：/mnt/data/map_complexity.csv 或 /mnt/data/features_vs_success.csv）
# - 自动推断目标列（优先：success_rate → Complexity → complexity → target → y）
# - 自动匹配特征列（含别名：Size/size、Agents/num_agents、Density/density、Density_actual/density_actual 等）
# - 对系数施加“有符号约束”（如 num_agents<=0、FRA/FDA<=0、Size>=0 等）
# - 拟合成功后，把预测值与残差两列写回到新 CSV：*_with_pred.csv（不覆盖原表）

# 需要依赖：cvxpy、numpy、pandas（若 OSQP 不可用会自动退回 ECOS/ECOS_BB）
# """

# import os
# import sys
# import json
# import numpy as np
# import pandas as pd

# # ----- 配置区域（按需修改） -----
# # 优先尝试的 CSV 路径（找到一个就用它）
# CSV_CANDIDATES = [
#     "C:/Users/MSc_SEIoT_1/MAPF_G2RL-main - train/train_gray3d-Copy-FDA.csv",
# ]

# # 目标列（None 表示自动推断：success_rate→Complexity→complexity→target→y）
# TARGET_COL = None

# # 候选特征（用“规范名”表示）；脚本会自动在 CSV 中找别名列
# CANDIDATE_FEATURES = [
#     "size", "num_agents", "density", "density_actual",
#     "LDD", "BN", "MC", "DLR", "FRA", "FPA"
# ]

# # 特征的符号约束（键用“规范名”）
# # ">=0": 系数非负；"<=0": 系数非正；"free": 不约束
# SIGN_CONSTRAINT = {
#     "size":            ">=0",
#     "num_agents":      ">=0",
#     "density":         ">=0",
#     "density_actual":  ">=0",
#     "LDD":             ">=0",
#     "BN":              ">=0",
#     "MC":              ">=0",
#     "DLR":             ">=0",
#     "FRA":             ">=0",
#     "FPA":             ">=0",
# }

# # L2 正则强度（0 表示不加正则）
# L2_LAMBDA = 1e-3

# # 输出文件名后缀
# OUTPUT_SUFFIX = "_with_pred.csv"
# WEIGHTS_JSON   = "_weights.json"
# # --------------------------------


# def _find_csv_path(candidates):
#     for p in candidates:
#         if os.path.exists(p):
#             return p
#     raise FileNotFoundError(
#         f"找不到 CSV。请把你的 CSV 放在这些路径之一：\n{candidates}"
#     )


# def _import_cvxpy():
#     try:
#         import cvxpy as cp
#         return cp
#     except Exception as e:
#         print("❌ 未能导入 cvxpy，请先安装： pip install cvxpy")
#         raise


# def _pick_target_col(df: pd.DataFrame, target_pref=None) -> str:
#     if target_pref and target_pref in df.columns:
#         return target_pref
#     for cand in ["success_rate", "Complexity", "complexity", "target", "y"]:
#         if cand in df.columns:
#             return cand
#     raise ValueError(
#         "自动选择目标列失败：未找到 success_rate/Complexity/complexity/target/y 中任意一个。"
#         "请把 TARGET_COL 改成表内存在的列名。"
#     )


# def _build_feature_mapping(df_cols):
#     """
#     构造“规范名→CSV实际列名”的映射（大小写与常见别名均可）。
#     规范名（右）: 可能的别名（左）
#     """
#     alias = {
#         "size":           ["size", "Size"],
#         "num_agents":     ["num_agents", "Agents", "agents", "n_agents"],
#         "density":        ["density", "Density"],
#         "density_actual": ["density_actual", "Density_actual", "actual_density"],
#         "LDD":            ["LDD", "ldd"],
#         "BN":             ["BN", "bn", "Bottleneck", "bottleneck"],
#         "MC":             ["MC", "mc", "MovingCost", "moving_cost"],
#         "DLR":            ["DLR", "dlr"],
#         "FRA":            ["FRA", "fra"],
#         "FPA":            ["FPA", "fpa"],
#     }

#     # 大小写不敏感匹配
#     lower_cols = {c.lower(): c for c in df_cols}
#     mapping = {}
#     for canonical, cands in alias.items():
#         hit = None
#         for c in cands:
#             if c in df_cols:
#                 hit = c
#                 break
#             if c.lower() in lower_cols:
#                 hit = lower_cols[c.lower()]
#                 break
#         if hit is not None:
#             mapping[canonical] = hit
#     return mapping


# def _solve_with_constraints(X, y, feature_names, sign_constraint, l2_lambda=0.0):
#     cp = _import_cvxpy()

#     n, d = X.shape
#     w = cp.Variable(d)   # 系数
#     b = cp.Variable()    # 截距

#     # 残差 & 损失
#     resid = X @ w + b - y
#     loss = cp.sum_squares(resid)
#     if l2_lambda and l2_lambda > 0:
#         loss += l2_lambda * cp.sum_squares(w)

#     # 施加符号约束
#     constraints = []
#     for j, name in enumerate(feature_names):
#         rule = sign_constraint.get(name, "free")
#         if rule == ">=0":
#             constraints += [w[j] >= 0]
#         elif rule == "<=0":
#             constraints += [w[j] <= 0]
#         # "free" 不加任何约束

#     prob = cp.Problem(cp.Minimize(loss), constraints)

#     # 优先 OSQP，不行就退回 ECOS/ECOS_BB
#     solvers_by_pref = [cp.OSQP, cp.ECOS, cp.ECOS_BB]
#     last_err = None
#     for solver in solvers_by_pref:
#         try:
#             prob.solve(solver=solver, verbose=True, max_iters=10000)
#             if prob.status in ("optimal", "optimal_inaccurate"):
#                 return w.value, b.value, prob.status
#         except Exception as e:
#             last_err = e
#             continue

#     raise RuntimeError(f"求解失败（已尝试 OSQP/ECOS/ECOS_BB）。最后错误：{last_err}")


# def main():
#     # 1) 选 CSV
#     csv_path = _find_csv_path(CSV_CANDIDATES)
#     print("📄 读取 CSV:", csv_path)
#     df = pd.read_csv(csv_path)
#     print("Columns:", list(df.columns))
#     print("Shape:", df.shape)

#     # 2) 选目标列
#     target_col = _pick_target_col(df, TARGET_COL)
#     print("🎯 使用目标列:", target_col)
#     y = df[target_col].astype(float).values  # shape (n,)

#     # 3) 特征选择（带别名）
#     mapping = _build_feature_mapping(df.columns)
#     # 只取表中真正存在的特征
#     used_features = [f for f in CANDIDATE_FEATURES if f in mapping]
#     if not used_features:
#         raise ValueError("❌ 没有任何候选特征在表中找到，请检查列名或 CANDIDATE_FEATURES。")
#     print("🧩 使用特征（规范名）:", used_features)
#     print("↳ 列映射（规范名→CSV列名）:", {k: mapping[k] for k in used_features})

#     # 4) 丢弃含 NaN 的行
#     cols_need = [mapping[f] for f in used_features] + [target_col]
#     dff = df[cols_need].dropna().copy()
#     if dff.empty:
#         raise ValueError("❌ 有效行数为 0（特征或目标存在 NaN）。请清洗数据后再试。")
#     print("✅ 可用样本数:", len(dff))

#     # 5) 组装 X/y
#     X = dff[[mapping[f] for f in used_features]].astype(float).values   # (n,d)
#     y_fit = dff[target_col].astype(float).values                        # (n,)

#     # 6) 求解带符号约束的线性回归
#     print("🚀 开始拟合（带符号约束）...")
#     w_val, b_val, status = _solve_with_constraints(
#         X, y_fit, used_features, SIGN_CONSTRAINT, L2_LAMBDA
#     )
#     print("Solver status:", status)
#     print("w:", w_val)
#     print("b:", float(b_val))

#     # 7) 预测与残差（在原表维度上计算，对 NaN 做跳过）
#     #    先按原表的列顺序构造 X_all（缺失行会是 NaN）
#     X_all = df[[mapping[f] for f in used_features]].astype(float)
#     pred = np.full(len(df), np.nan, dtype=float)

#     # 只对完整行做预测
#     mask_ok = ~X_all.isna().any(axis=1)
#     pred[mask_ok.values] = X_all.loc[mask_ok].values @ w_val + float(b_val)

#     # 残差（只在 y 存在且 pred 已计算出来的行有效）
#     residual = np.full(len(df), np.nan, dtype=float)
#     mask_resid = mask_ok & df[target_col].notna()
#     residual[mask_resid.values] = df.loc[mask_resid, target_col].astype(float).values - pred[mask_resid.values]

#     # 写回两列
#     pred_col = f"pred_{target_col}"
#     resid_col = f"residual_{target_col}"
#     df[pred_col] = pred
#     df[resid_col] = residual

#     # 8) 保存输出
#     base, ext = os.path.splitext(csv_path)
#     out_csv = base + OUTPUT_SUFFIX
#     df.to_csv(out_csv, index=False, encoding="utf-8-sig")
#     print(f"✅ 已写出：{out_csv}")

#     # 9) 顺便把系数也存一份 JSON
#     weights = {
#         "intercept": float(b_val),
#         "weights": {feat: float(w_val[i]) for i, feat in enumerate(used_features)},
#         "target": target_col,
#         "features_used": used_features,
#         "sign_constraint": {k: SIGN_CONSTRAINT.get(k, "free") for k in used_features},
#         "l2_lambda": L2_LAMBDA,
#         "status": status,
#     }
#     weights_path = base + WEIGHTS_JSON
#     with open(weights_path, "w", encoding="utf-8") as f:
#         json.dump(weights, f, ensure_ascii=False, indent=2)
#     print(f"📝 系数已保存：{weights_path}")

#     # 10) 打印一版可读的系数表
#     print("\n==== Learned Coefficients ====")
#     for k in used_features:
#         print(f"{k:>16s} : {weights['weights'][k]: .6f}")
#     print(f"{'intercept':>16s} : {weights['intercept']: .6f}")
#     print("==============================")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Constrained linear regression with sign constraints (cvxpy).
- 自动读取 CSV（默认：/mnt/data/map_complexity.csv 或 /mnt/data/features_vs_success.csv）
- 自动推断目标列（优先：success_rate → Complexity → complexity → target → y）
- 自动匹配特征列（含别名：Size/size、Agents/num_agents、Density/density、Density_actual/density_actual 等）
- 对系数施加“有符号约束”（如 num_agents<=0、FRA/FDA<=0、Size>=0 等）
- 拟合成功后，把预测值与残差两列写回到新 CSV：*_with_pred.csv（不覆盖原表）
- 评估：Pearson R、R^2、MAE、RMSE、MAPE，并生成诊断图

需要依赖：cvxpy、numpy、pandas、matplotlib、scikit-learn（若 OSQP 不可用会自动退回 ECOS/ECOS_BB）
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ----- 配置区域（按需修改） -----
CSV_CANDIDATES = [
    "C:/Users/MSc_SEIoT_1/MAPF_G2RL-main - train/train_gray3d-Copy-FDA.csv",
]

TARGET_COL = None

CANDIDATE_FEATURES = [
    "size", "num_agents", "density", "density_actual",
    "LDD", "BN", "MC", "DLR", "FRA", "FPA"
]

SIGN_CONSTRAINT = {
    "size":            ">=0",
    "num_agents":      ">=0",
    "density":         ">=0",
    "density_actual":  ">=0",
    "LDD":             ">=0",
    "BN":              ">=0",
    "MC":              ">=0",
    "DLR":             ">=0",
    "FRA":             ">=0",
    "FPA":             ">=0",
}

L2_LAMBDA = 1e-3

OUTPUT_SUFFIX = "_with_pred.csv"
WEIGHTS_JSON   = "_weights.json"
FIG_PRED_ACT   = "_pred_vs_actual.png"
FIG_RESID_HIST = "_residual_hist.png"
FIG_RESID_FIT  = "_residuals_vs_fitted.png"
# --------------------------------


def _find_csv_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"找不到 CSV。请把你的 CSV 放在这些路径之一：\n{candidates}"
    )


def _import_cvxpy():
    try:
        import cvxpy as cp
        return cp
    except Exception as e:
        print("❌ 未能导入 cvxpy，请先安装： pip install cvxpy")
        raise


def _pick_target_col(df: pd.DataFrame, target_pref=None) -> str:
    if target_pref and target_pref in df.columns:
        return target_pref
    for cand in ["success_rate", "Complexity", "complexity", "target", "y"]:
        if cand in df.columns:
            return cand
    raise ValueError(
        "自动选择目标列失败：未找到 success_rate/Complexity/complexity/target/y 中任意一个。"
        "请把 TARGET_COL 改成表内存在的列名。"
    )


def _build_feature_mapping(df_cols):
    alias = {
        "size":           ["size", "Size"],
        "num_agents":     ["num_agents", "Agents", "agents", "n_agents"],
        "density":        ["density", "Density"],
        "density_actual": ["density_actual", "Density_actual", "actual_density"],
        "LDD":            ["LDD", "ldd"],
        "BN":             ["BN", "bn", "Bottleneck", "bottleneck"],
        "MC":             ["MC", "mc", "MovingCost", "moving_cost"],
        "DLR":            ["DLR", "dlr"],
        "FRA":            ["FRA", "fra"],
        "FPA":            ["FPA", "fpa"],
    }
    lower_cols = {c.lower(): c for c in df_cols}
    mapping = {}
    for canonical, cands in alias.items():
        hit = None
        for c in cands:
            if c in df_cols:
                hit = c
                break
            if c.lower() in lower_cols:
                hit = lower_cols[c.lower()]
                break
        if hit is not None:
            mapping[canonical] = hit
    return mapping


def _solve_with_constraints(X, y, feature_names, sign_constraint, l2_lambda=0.0):
    cp = _import_cvxpy()

    n, d = X.shape
    w = cp.Variable(d)   # 系数
    b = cp.Variable()    # 截距

    resid = X @ w + b - y
    loss = cp.sum_squares(resid)
    if l2_lambda and l2_lambda > 0:
        loss += l2_lambda * cp.sum_squares(w)

    constraints = []
    for j, name in enumerate(feature_names):
        rule = sign_constraint.get(name, "free")
        if rule == ">=0":
            constraints += [w[j] >= 0]
        elif rule == "<=0":
            constraints += [w[j] <= 0]

    prob = cp.Problem(cp.Minimize(loss), constraints)

    solvers_by_pref = [cp.OSQP, cp.ECOS, cp.ECOS_BB]
    last_err = None
    for solver in solvers_by_pref:
        try:
            prob.solve(solver=solver, verbose=False, max_iters=10000)
            if prob.status in ("optimal", "optimal_inaccurate"):
                return w.value, b.value, prob.status
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"求解失败（已尝试 OSQP/ECOS/ECOS_BB）。最后错误：{last_err}")


def main():
    # 1) 选 CSV
    csv_path = _find_csv_path(CSV_CANDIDATES)
    print("📄 读取 CSV:", csv_path)
    df = pd.read_csv(csv_path)
    print("Columns:", list(df.columns))
    print("Shape:", df.shape)

    # 2) 选目标列
    target_col = _pick_target_col(df, TARGET_COL)
    print("🎯 使用目标列:", target_col)
    y = df[target_col].astype(float).values  # shape (n,)

    # 3) 特征选择（带别名）
    mapping = _build_feature_mapping(df.columns)
    used_features = [f for f in CANDIDATE_FEATURES if f in mapping]
    if not used_features:
        raise ValueError("❌ 没有任何候选特征在表中找到，请检查列名或 CANDIDATE_FEATURES。")
    print("🧩 使用特征（规范名）:", used_features)
    print("↳ 列映射（规范名→CSV列名）:", {k: mapping[k] for k in used_features})

    # 4) 丢弃含 NaN 的行
    cols_need = [mapping[f] for f in used_features] + [target_col]
    dff = df[cols_need].dropna().copy()
    if dff.empty:
        raise ValueError("❌ 有效行数为 0（特征或目标存在 NaN）。请清洗数据后再试。")
    print("✅ 可用样本数:", len(dff))

    # 5) 组装 X/y
    X = dff[[mapping[f] for f in used_features]].astype(float).values   # (n,d)
    y_fit = dff[target_col].astype(float).values                        # (n,)

    # 6) 求解带符号约束的线性回归
    print("🚀 开始拟合（带符号约束）...")
    w_val, b_val, status = _solve_with_constraints(
        X, y_fit, used_features, SIGN_CONSTRAINT, L2_LAMBDA
    )
    print("Solver status:", status)
    print("w:", w_val)
    print("b:", float(b_val))

    # 7) 预测与残差（在原表维度上计算，对 NaN 做跳过）
    X_all = df[[mapping[f] for f in used_features]].astype(float)
    pred = np.full(len(df), np.nan, dtype=float)
    mask_ok = ~X_all.isna().any(axis=1)
    pred[mask_ok.values] = X_all.loc[mask_ok].values @ w_val + float(b_val)

    residual = np.full(len(df), np.nan, dtype=float)
    mask_resid = mask_ok & df[target_col].notna()
    residual[mask_resid.values] = df.loc[mask_resid, target_col].astype(float).values - pred[mask_resid.values]

    pred_col = f"pred_{target_col}"
    resid_col = f"residual_{target_col}"
    df[pred_col] = pred
    df[resid_col] = residual

    # 8) 保存输出 CSV
    base, ext = os.path.splitext(csv_path)
    out_csv = base + OUTPUT_SUFFIX
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 已写出：{out_csv}")

    # ======== 9) === 评估与可视化（基于“参与拟合的样本”） ========
    # 仅对 dff（训练参与的无缺失样本）计算评估指标和作图
    y_pred_fit = X @ w_val + float(b_val)

    # 指标
    # Pearson 相关系数 R
    if len(y_fit) >= 2 and np.std(y_fit) > 0 and np.std(y_pred_fit) > 0:
        R = float(np.corrcoef(y_fit, y_pred_fit)[0, 1])
    else:
        R = float("nan")
    R2 = float(r2_score(y_fit, y_pred_fit))
    MAE = float(mean_absolute_error(y_fit, y_pred_fit))
    RMSE = float(np.sqrt(mean_squared_error(y_fit, y_pred_fit)))
    # 为避免除零，对接近 0 的 y 进行保护
    denom = np.where(np.abs(y_fit) < 1e-12, 1e-12, np.abs(y_fit))
    MAPE = float(np.mean(np.abs((y_fit - y_pred_fit) / denom)) * 100.0)

    print("\n==== Metrics (on training/fit subset) ====")
    print(f"Pearson R : {R: .6f}")
    print(f"R^2       : {R2: .6f}")
    print(f"MAE       : {MAE: .6f}")
    print(f"RMSE      : {RMSE: .6f}")
    print(f"MAPE (%)  : {MAPE: .2f}")
    print("==========================================\n")

    # 图 1：Predicted vs. Actual（带 y=x 参考线）
    fig1 = base + FIG_PRED_ACT
    plt.figure()
    plt.scatter(y_fit, y_pred_fit, alpha=0.7)
    minv = float(min(np.min(y_fit), np.min(y_pred_fit)))
    maxv = float(max(np.max(y_fit), np.max(y_pred_fit)))
    plt.plot([minv, maxv], [minv, maxv], linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs. Actual\nR={R:.3f}, R²={R2:.3f}, MAE={MAE:.3f}, RMSE={RMSE:.3f}")
    plt.tight_layout()
    plt.savefig(fig1, dpi=160)
    plt.close()
    print(f"🖼 已保存图：{fig1}")

    # 图 2：Residual Histogram
    fig2 = base + FIG_RESID_HIST
    plt.figure()
    plt.hist(y_fit - y_pred_fit, bins=30)
    plt.xlabel("Residual (y - y_pred)")
    plt.ylabel("Count")
    plt.title("Residuals Histogram")
    plt.tight_layout()
    plt.savefig(fig2, dpi=160)
    plt.close()
    print(f"🖼 已保存图：{fig2}")

    # 图 3：Residuals vs. Fitted
    fig3 = base + FIG_RESID_FIT
    plt.figure()
    plt.scatter(y_pred_fit, y_fit - y_pred_fit, alpha=0.7)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Fitted (Predicted)")
    plt.ylabel("Residual")
    plt.title("Residuals vs. Fitted")
    plt.tight_layout()
    plt.savefig(fig3, dpi=160)
    plt.close()
    print(f"🖼 已保存图：{fig3}")

    # 10) 保存系数 JSON（附加评估指标）
    weights = {
        "intercept": float(b_val),
        "weights": {feat: float(w_val[i]) for i, feat in enumerate(used_features)},
        "target": target_col,
        "features_used": used_features,
        "sign_constraint": {k: SIGN_CONSTRAINT.get(k, "free") for k in used_features},
        "l2_lambda": L2_LAMBDA,
        "status": status,
        "metrics_on_fit": {
            "pearson_r": R,
            "r2": R2,
            "mae": MAE,
            "rmse": RMSE,
            "mape_percent": MAPE
        },
        "diagnostic_figures": {
            "pred_vs_actual": fig1,
            "residual_hist": fig2,
            "residuals_vs_fitted": fig3
        }
    }
    weights_path = base + WEIGHTS_JSON
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)
    print(f"📝 系数与指标已保存：{weights_path}")

    # 11) 打印一版可读的系数表
    print("\n==== Learned Coefficients ====")
    for k in used_features:
        print(f"{k:>16s} : {weights['weights'][k]: .6f}")
    print(f"{'intercept':>16s} : {weights['intercept']: .6f}")
    print("==============================")

if __name__ == "__main__":
    main()

