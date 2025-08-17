
from __future__ import annotations
import os
import glob
import math
from typing import Any, Dict, Optional, Tuple, Union

import yaml
import numpy as np
import pandas as pd


def compute_map_complexity(
    spec: Union[str, Dict[str, Any], np.ndarray, list],
    *,
    intercept: float,
    weights: Dict[str, float],
    feature_mean_std: Optional[Dict[str, Tuple[float, float]]] = None,
    size_mode: str = "max",
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """Compute complexity for a single map spec.

    Parameters
    ----------
    spec : str | dict | 2D array-like
        - str: path to a YAML file
        - dict: already-loaded YAML mapping
        - array-like: 2D grid where 1=obstacle, 0=free
    intercept : float
        Linear model intercept 'b' from your trained formula.
    weights : dict[str, float]
        Linear model weights mapping feature name -> coefficient.
        Common keys: 'Size', 'Agents', 'Density', optionally 'LDD','BN','MC','DLR', etc.
    feature_mean_std : dict[str, (mean, std)], optional
        If your model used standardized features, provide the same mean/std per feature.
    size_mode : str
        How to derive Size if height/width are present and 'size' is missing:
        - 'max': max(height, width)
        - 'area': height * width
        - 'diag': sqrt(height**2 + width**2)

    Returns
    -------
    complexity : float
        Computed complexity (intercept + sum(w_i * x_i)).
    used_features : dict[str, float]
        Transformed feature values actually used (after standardization if provided).
    raw_features : dict[str, float]
        Raw, untransformed features extracted from the spec.
    """
    # Load YAML if needed
    if isinstance(spec, str):
        with open(spec, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif isinstance(spec, (np.ndarray, list)):
        data = {"grid": np.asarray(spec)}
    elif isinstance(spec, dict):
        data = dict(spec)
    else:
        raise TypeError("spec must be a file path, dict, or 2D array-like")

    # Extract features
    raw = extract_basic_features(data, size_mode=size_mode)
    raw.update(compute_optional_features(data))

    # Apply model
    means = feature_mean_std or {}
    used: Dict[str, float] = {}
    z = float(intercept)
    for name, w in weights.items():
        if name not in raw:
            continue
        x = raw[name]
        if name in means:
            mu, sigma = means[name]
            if sigma and sigma != 0:
                x = (x - mu) / sigma
            else:
                x = 0.0
        used[name] = float(x)
        z += float(w) * float(x)

    return float(z), used, raw


def compute_map_complexity_for_dir(
    in_dir: str,
    *,
    intercept: float,
    weights: Dict[str, float],
    feature_mean_std: Optional[Dict[str, Tuple[float, float]]] = None,
    pattern: str = "*.yaml",
    size_mode: str = "max",
    write_back: bool = False,
    out_key: str = "complexity",
) -> pd.DataFrame:
    """Batch-compute complexity for all YAML maps in a directory."""
    rows = []
    paths = sorted(glob.glob(os.path.join(in_dir, pattern)))
    for p in paths:
        try:
            cpx, used, raw = compute_map_complexity(
                p,
                intercept=intercept,
                weights=weights,
                feature_mean_std=feature_mean_std,
                size_mode=size_mode,
            )
            rows.append({
                "file": os.path.basename(p),
                "path": p,
                "complexity": cpx,
                **{f"raw_{k}": v for k, v in raw.items()},
                **{f"used_{k}": v for k, v in used.items()},
            })

            if write_back:
                with open(p, "r", encoding="utf-8") as f:
                    y = yaml.safe_load(f) or {}
                if "metadata" not in y or not isinstance(y["metadata"], dict):
                    y["metadata"] = {}
                y["metadata"][out_key] = float(cpx)
                with open(p, "w", encoding="utf-8") as f:
                    yaml.safe_dump(y, f, allow_unicode=True, sort_keys=False)

        except Exception as e:
            rows.append({"file": os.path.basename(p), "path": p, "error": str(e)})
    return pd.DataFrame(rows)


def extract_basic_features(y: Dict[str, Any], size_mode: str = "max") -> Dict[str, float]:
    feats: Dict[str, float] = {}

    size_val = infer_size(y, mode=size_mode)
    if size_val is not None:
        feats["Size"] = float(size_val)

    agents_val = infer_agents(y)
    if agents_val is not None:
        feats["Agents"] = float(agents_val)

    dens = infer_density_from_params(y)
    if dens is None:
        dens = infer_density_from_grid(y)
    if dens is not None:
        feats["Density"] = float(dens)

    return feats


def compute_optional_features(y: Dict[str, Any]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for k in ("LDD", "BN", "MC", "DLR"):
        if k in y and isinstance(y[k], (int, float)):
            feats[k] = float(y[k])
    return feats


def infer_size(y: Dict[str, Any], mode: str = "max") -> Optional[float]:
    if "size" in y and isinstance(y["size"], (int, float)):
        return float(y["size"])
    h = y.get("height", None)
    w = y.get("width", None)
    if isinstance(h, (int, float)) and isinstance(w, (int, float)):
        if mode == "max":
            return float(max(h, w))
        if mode == "area":
            return float(h) * float(w)
        if mode == "diag":
            return float(math.sqrt(h*h + w*w))
        raise ValueError("size_mode must be 'max'|'area'|'diag'")
    return None


def infer_agents(y: Dict[str, Any]) -> Optional[float]:
    for k in ("num_agents", "agents", "agent_count"):
        if k in y and isinstance(y[k], (int, float)):
            return float(y[k])
    return None


def infer_density_from_params(y: Dict[str, Any]) -> Optional[float]:
    for k in ("density", "obstacle_density"):
        if k in y and isinstance(y[k], (int, float)):
            return float(y[k])
    return None


def infer_density_from_grid(y: Dict[str, Any]) -> Optional[float]:
    grid = y.get("grid", None)
    if grid is None:
        grid = y.get("map", None)

    if grid is None:
        return None

    # numpy 数组
    if isinstance(grid, np.ndarray):
        total = grid.size
        obstacle = int((grid == 1).sum())  # 1=障碍
        return obstacle / total if total > 0 else None

    # python list[list[int]]
    if isinstance(grid, list) and len(grid) > 0 and isinstance(grid[0], list):
        arr = np.array(grid)
        total = arr.size
        obstacle = int((arr == 1).sum())
        return obstacle / total if total > 0 else None

    obstacles = y.get("obstacles")
    if isinstance(obstacles, list):
        h = y.get("height") or y.get("H") or y.get("rows")
        w = y.get("width") or y.get("W") or y.get("cols")
        s = y.get("size")
        if s and not (h and w):
            h = w = s
        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
            total = h * w
            return len(obstacles) / total
    return None


def cli():
    import argparse
    parser = argparse.ArgumentParser(description="Compute map complexity from YAMLs.")
    parser.add_argument("--in_dir", type=str, required=True, help="Directory containing YAML maps")
    parser.add_argument("--pattern", type=str, default="*.yaml", help="Glob pattern for YAML files")
    parser.add_argument("--write_back", action="store_true", help="Write complexity into YAML metadata")
    parser.add_argument("--out_csv", type=str, default="map_complexity.csv", help="Output CSV path")
    parser.add_argument("--size_mode", type=str, default="max", choices=["max", "area", "diag"])
    parser.add_argument("--weights_yaml", type=str, help="YAML with {'intercept': b, 'weights': {...}}")
    parser.add_argument("--meanstd_yaml", type=str, help="YAML with {'feature_mean_std': {name: [mean, std]}}")

    args = parser.parse_args()

    intercept = 0.0
    weights = {}
    feat_ms = None

    if args.weights_yaml:
        with open(args.weights_yaml, "r", encoding="utf-8") as f:
            wcfg = yaml.safe_load(f) or {}
        intercept = float(wcfg.get("intercept", 0.0))
        weights = {str(k): float(v) for k, v in (wcfg.get("weights") or {}).items()}

    if args.meanstd_yaml:
        with open(args.meanstd_yaml, "r", encoding="utf-8") as f:
            mscfg = yaml.safe_load(f) or {}
        raw = mscfg.get("feature_mean_std") or {}
        feat_ms = {str(k): (float(v[0]), float(v[1])) for k, v in raw.items()}

    df = compute_map_complexity_for_dir(
        args.in_dir,
        intercept=intercept,
        weights=weights,
        feature_mean_std=feat_ms,
        pattern=args.pattern,
        size_mode=args.size_mode,
        write_back=args.write_back,
    )
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved: {args.out_csv}")
    if args.write_back:
        print("📝 complexity written to metadata.complexity in each YAML.")
