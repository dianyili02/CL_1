
"""
complexity.py
=============

A runnable module to compute MAPF-oriented map complexity metrics, including:
- Local obstacle density distribution (LDD)
- Bottleneck coefficient (BN)
- Map connectivity (MC)
- Potential deadlock risk (DLR)
plus the basic features: map size, agent count, global obstacle density.

Dependencies: numpy, scipy, scikit-image, networkx

Main API:
    compute_complexity(grid, starts=None, goals=None, K=30, weights=None, window=None,
                       connectivity=4, random_state=None, max_astar_steps=10000)

Input:
    grid: 2D numpy array with 0 = free, 1 = obstacle
    starts/goals: Optional list of (x,y) tuples (row, col). If provided and same length,
                  DLR will use these pairs. Otherwise, K random pairs are sampled.
    K: number of random start/goal pairs when starts/goals not provided.
    weights: dict of weights for top-level combination, or None for default.
    window: local density window size (odd int), default auto based on grid size.
    connectivity: 4 or 8 for grid graph adjacency.
    random_state: int seed or np.random.Generator for reproducibility.
    max_astar_steps: guard for A* to avoid pathological loops.

Returns: dict with per-metric values and a combined "Complexity" score.
"""

from __future__ import annotations

import math
import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import networkx as nx

from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt

from skimage.measure import label, regionprops


Array2D = np.ndarray
Coord = Tuple[int, int]


# ------------------------- Utility & normalization -------------------------

def _rng_from(random_state=None):
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def robust_minmax(x: float, lo: float, hi: float) -> float:
    """Map x to [0,1] given expected range [lo, hi]. Clipped."""
    if hi == lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def zcombine(values: Iterable[float], weights: Optional[Iterable[float]] = None) -> float:
    """Weighted sum after simple clipping to [0,1]."""
    vals = np.asarray(list(values), dtype=float)
    if weights is None:
        weights = np.ones_like(vals) / (len(vals) if len(vals) else 1.0)
    w = np.asarray(list(weights), dtype=float)
    if len(vals) == 0:
        return 0.0
    w = w / (w.sum() if w.sum() != 0 else 1.0)
    return float(np.clip(np.dot(vals, w), 0.0, 1.0))


def grid_shape_size(grid: Array2D) -> Tuple[int, int, int]:
    assert grid.ndim == 2, "grid must be 2D"
    h, w = grid.shape
    return h, w, h * w


def free_mask(grid: Array2D) -> Array2D:
    return (grid == 0).astype(np.uint8)


def global_density(grid: Array2D) -> float:
    h, w, n = grid_shape_size(grid)
    return float(grid.sum() / n)


def _auto_window(grid: Array2D) -> int:
    h, w, _ = grid_shape_size(grid)
    size = max(h, w)
    win = max(5, min(15, size // 8))
    if win % 2 == 0:
        win += 1
    return int(win)


# ------------------ 1) Local Obstacle Density Distribution (LDD) ------------------

def local_density_map(grid: Array2D, window: Optional[int] = None) -> Array2D:
    """Compute local obstacle density via uniform filter over a sliding window."""
    if window is None:
        window = _auto_window(grid)
    # Use uniform filter on obstacle mask (1=obstacle)
    density = ndi.uniform_filter(grid.astype(float), size=window, mode="nearest")
    return density  # values in [0,1]


def ldd_features(grid: Array2D, window: Optional[int] = None, bins: int = 20) -> Dict[str, float]:
    rho = local_density_map(grid, window=window)
    var = float(np.var(rho))
    mean = float(np.mean(rho))
    std = float(np.std(rho))
    cv = float(std / (mean + 1e-6))
    hist, _ = np.histogram(rho, bins=bins, range=(0, 1), density=True)
    entropy = float(-(hist * np.log(hist + 1e-12)).sum())
    # Normalize with loose expected ranges
    LDD_var = robust_minmax(var, 0.0, 0.25)     # var roughly in [0, 0.25]
    LDD_cv = robust_minmax(cv, 0.0, 2.0)        # cv in [0,2] loosely
    LDD_entropy = robust_minmax(entropy, 0.0, math.log(bins))  # max entropy ~ log(bins)
    LDD = zcombine([LDD_var, LDD_cv, LDD_entropy])
    return {
        "LDD": LDD,
        "LDD_var": var,
        "LDD_cv": cv,
        "LDD_entropy": entropy,
    }


# ------------------ Graph construction helpers ------------------

def neighbors_4(r: int, c: int, h: int, w: int):
    if r > 0: yield (r - 1, c)
    if r + 1 < h: yield (r + 1, c)
    if c > 0: yield (r, c - 1)
    if c + 1 < w: yield (r, c + 1)


def neighbors_8(r: int, c: int, h: int, w: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                yield (nr, nc)


def grid_to_graph(grid: Array2D, connectivity: int = 4) -> nx.Graph:
    """Convert free cells to a graph with 4- or 8-neighborhood connectivity."""
    fm = free_mask(grid)
    h, w = fm.shape
    G = nx.Graph()
    # add nodes
    free_coords = np.argwhere(fm == 1)
    for r, c in free_coords:
        G.add_node((int(r), int(c)))
    # add edges
    ngh = neighbors_4 if connectivity == 4 else neighbors_8
    for r, c in free_coords:
        for nr, nc in ngh(int(r), int(c), h, w):
            if fm[nr, nc] == 1:
                G.add_edge((int(r), int(c)), (int(nr), int(nc)))
    return G


def largest_component(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G.copy()
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    gcc_nodes = components[0]
    return G.subgraph(gcc_nodes).copy()


# ------------------ Corridor widths & distance transform ------------------

def corridor_widths(grid: Array2D) -> np.ndarray:
    """
    Approximate corridor width at free cells using distance transform.
    Width ~ 2 * distance_to_nearest_obstacle.
    """
    fm = free_mask(grid)
    # distance transform on free space: distance to nearest obstacle (on inverse mask)
    # Use EDT for Euclidean distance, could use cityblock via scipy if desired.
    dist = distance_transform_edt(fm)  # distance to nearest obstacle cell (which is 0 in fm)
    widths = 2.0 * dist  # approximate corridor width
    return widths


# ------------------ 2) Bottleneck Coefficient (BN) ------------------

def bottleneck_features(grid: Array2D, connectivity: int = 4) -> Dict[str, float]:
    widths = corridor_widths(grid)
    fm = free_mask(grid)
    width_vals = widths[fm == 1]
    if width_vals.size == 0:
        return {"BN": 0.0, "BN_width": 1.0, "BN_art": 0.0, "BN_spec": 1.0}
    w10 = float(np.percentile(width_vals, 10))
    # Normalize: narrower widths => harder. Expect widths in [0, ~min(h,w)]
    h, w, _ = grid_shape_size(grid)
    max_w = float(min(h, w))
    BN_width = robust_minmax((max_w - w10), 0.0, max_w)  # invert & normalize

    # Articulation points
    G = grid_to_graph(grid, connectivity=connectivity)
    n_nodes = max(1, G.number_of_nodes())
    if n_nodes <= 2:
        BN_art = 0.0
        BN_spec = 0.0
    else:
        try:
            arts = list(nx.articulation_points(G))
            BN_art = len(arts) / n_nodes
        except Exception:
            BN_art = 0.0

        # Spectral: algebraic connectivity (lambda2) on GCC
        GCC = largest_component(G)
        try:
            lambda2 = nx.algebraic_connectivity(GCC, tol=1e-4, method="tracemin_pcg")
            BN_spec = robust_minmax((1.0 / (lambda2 + 1e-6)), 0.0, 10.0)
        except Exception:
            BN_spec = 0.0

    BN = zcombine([BN_width, BN_art, BN_spec])
    return {
        "BN": BN,
        "BN_width": float(w10),
        "BN_art": float(BN_art),
        "BN_spec": float(BN_spec),
    }


# ------------------ 3) Map Connectivity (MC) ------------------

def connectivity_features(grid: Array2D, connectivity: int = 4, sample_pairs: int = 200, random_state=None) -> Dict[str, float]:
    G = grid_to_graph(grid, connectivity=connectivity)
    n = G.number_of_nodes()
    if n == 0:
        return {"MC": 1.0, "MC_comp": 1.0, "MC_lambda2": 1.0, "MC_aspl": 1.0}

    GCC = largest_component(G)
    gcc_n = GCC.number_of_nodes()
    MC_comp_raw = 1.0 - (gcc_n / n)  # fragmentation
    MC_comp = robust_minmax(MC_comp_raw, 0.0, 1.0)

    # lambda2 on GCC
    try:
        lambda2 = nx.algebraic_connectivity(GCC, tol=1e-4, method="tracemin_pcg")
        MC_lambda2 = robust_minmax((1.0 / (lambda2 + 1e-6)), 0.0, 10.0)
    except Exception:
        MC_lambda2 = 0.0

    # Average shortest path length on GCC (approx via sampling if big)
    rng = _rng_from(random_state)
    nodes = list(GCC.nodes())
    m = len(nodes)
    if m <= 1:
        aspl_raw = 0.0
    else:
        S = min(sample_pairs, m)
        # sample distinct pairs
        total = 0.0
        cnt = 0
        idx = rng.choice(m, size=S, replace=False) if S < m else np.arange(m)
        for i in idx:
            # BFS distances from nodes[i] for efficiency
            lengths = nx.single_source_shortest_path_length(GCC, nodes[i])
            # sample a few targets
            targets = rng.choice(m, size=min(10, m), replace=False)
            for j in targets:
                if i == j: 
                    continue
                d = lengths.get(nodes[j], None)
                if d is not None:
                    total += d
                    cnt += 1
        aspl_raw = (total / cnt) if cnt > 0 else float('inf')

    # Normalize ASPL by grid diameter ~ h+w
    h, w, _ = grid_shape_size(grid)
    dia = h + w
    MC_aspl = robust_minmax(aspl_raw, 0.0, max(1.0, dia))

    MC = zcombine([MC_comp, MC_lambda2, MC_aspl])
    return {
        "MC": MC,
        "MC_comp": float(MC_comp_raw),
        "MC_lambda2": float(MC_lambda2),
        "MC_aspl": float(MC_aspl),
    }


# ------------------ 4) Potential Deadlock Risk (DLR) ------------------

def detect_pockets_via_deadends(grid: Array2D, min_size: int = 6, connectivity: int = 4) -> int:
    """
    Heuristic: Count cul-de-sac (dead-end) basins of size >= min_size.
    We identify degree-1 nodes in the free-space graph and expand until a junction or boundary.
    """
    G = grid_to_graph(grid, connectivity=connectivity)
    if G.number_of_nodes() == 0:
        return 0
    degree = dict(G.degree())
    dead_ends = [n for n, d in degree.items() if d == 1]
    visited = set()
    pocket_count = 0
    for s in dead_ends:
        if s in visited:
            continue
        # expand along the corridor until a junction (deg>=3) or loop
        path = [s]
        cur = s
        prev = None
        visited.add(s)
        while True:
            nbrs = [v for v in G.neighbors(cur) if v != prev]
            if len(nbrs) != 1:
                # stop on junction or dead stop
                break
            nxt = nbrs[0]
            path.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt
            if G.degree(cur) >= 3:
                break
        # If corridor leads to a bulbous area (more than a few nodes attached), count as pocket
        # crude: BFS area beyond `cur` excluding the path back
        frontier = [cur]
        seen = set(path)  # block the corridor backward
        area = 0
        while frontier:
            u = frontier.pop()
            if u in seen:
                continue
            seen.add(u)
            area += 1
            for v in G.neighbors(u):
                if v not in seen:
                    frontier.append(v)
        if area >= min_size:
            pocket_count += 1
    return pocket_count


# def astar(grid: Array2D, start: Coord, goal: Coord, connectivity: int = 4, max_steps: int = 10000) -> List[Coord]:
#     """A* on grid, returns path including start and goal; [] if no path."""
#     h, w = grid.shape
#     fm = free_mask(grid)
#     if fm[start] == 0 or fm[goal] == 0:
#         return []

def astar(grid: Array2D, start: Coord, goal: Coord, connectivity: int = 4, max_steps: int = 10000) -> List[Coord]:
    """A* on grid, returns path including start and goal; [] if no path."""
    # 确保 start/goal 是 tuple(int,int)
    start = tuple(map(int, start))
    goal = tuple(map(int, goal))

    h, w = grid.shape
    fm = free_mask(grid)
    if fm[start] == 0 or fm[goal] == 0:
        return []


    ngh = neighbors_4 if connectivity == 4 else neighbors_8

    def heuristic(a: Coord, b: Coord) -> float:
        # Manhattan or Chebyshev depending on connectivity
        if connectivity == 4:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        else:
            return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    open_set = [(0 + heuristic(start, goal), 0, start, None)]  # (f, g, node, parent)
    came_from = {}
    gscore = {start: 0}
    visited = 0

    while open_set:
        f, g, current, parent = heapq.heappop(open_set)
        if current in came_from:
            continue
        came_from[current] = parent
        if current == goal:
            break
        visited += 1
        if visited > max_steps:
            break
        for nr, nc in ngh(*current, h, w):
            if fm[nr, nc] == 0:
                continue
            ng = g + 1
            n = (nr, nc)
            if ng < gscore.get(n, 1e18):
                gscore[n] = ng
                heapq.heappush(open_set, (ng + heuristic(n, goal), ng, n, current))

    # reconstruct
    if goal not in came_from:
        return []
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path


def simulate_conflicts(paths: List[List[Coord]]) -> Tuple[int, int, int]:
    """Count vertex and edge conflicts across synchronous execution, and steps on narrow cells.
    Returns (vertex_conflicts, edge_conflicts, total_steps).
    """
    if not paths:
        return 0, 0, 0
    T = max((len(p) for p in paths), default=0)
    vertex_conflicts = 0
    edge_conflicts = 0
    total_steps = 0

    # Expand last position if agent arrives earlier
    expanded = []
    for p in paths:
        if not p:
            expanded.append([])
            continue
        last = p[-1]
        q = p + [last] * (T - len(p))
        expanded.append(q)

    for t in range(T):
        positions = [p[t] for p in expanded if p]
        total_steps += len(positions)
        # vertex conflicts: same cell, same time
        counts = {}
        for pos in positions:
            counts[pos] = counts.get(pos, 0) + 1
        vertex_conflicts += sum(c - 1 for c in counts.values() if c > 1)

        # edge conflicts: (u->v) and (v->u) at same t
        if t + 1 < T:
            moves = [(p[t], p[t + 1]) for p in expanded if p]
            # put in set for quick lookup
            move_set = set(moves)
            for u, v in moves:
                if (v, u) in move_set and u != v:
                    edge_conflicts += 1
            # Each crossing counts twice in above scan; halve it.
            edge_conflicts //= 2

    return int(vertex_conflicts), int(edge_conflicts), int(total_steps)


def deadlock_features(
    grid: Array2D,
    starts: Optional[List[Coord]] = None,
    goals: Optional[List[Coord]] = None,
    K: int = 30,
    connectivity: int = 4,
    random_state=None,
    max_astar_steps: int = 10000,
) -> Dict[str, float]:
    rng = _rng_from(random_state)
    fm = free_mask(grid)
    free_coords = np.argwhere(fm == 1)
    nfree = len(free_coords)

    # Corridor & widths
    widths = corridor_widths(grid)
    narrow_mask = (widths <= 2.0) & (fm == 1)
    DL_corridor_raw = narrow_mask.sum() / max(1, nfree)
    DL_corridor = robust_minmax(DL_corridor_raw, 0.0, 0.5)

    # Articulation "load"
    G = grid_to_graph(grid, connectivity=connectivity)
    try:
        arts = list(nx.articulation_points(G))
        if len(arts) > 0:
            deg_vals = [G.degree(v) for v in arts]
            DL_artload_raw = float(np.mean(deg_vals))
        else:
            DL_artload_raw = 0.0
    except Exception:
        DL_artload_raw = 0.0
    # normalize by 4-neigh baseline 4 (or 8)
    norm_deg = 4.0 if connectivity == 4 else 8.0
    DL_artload = robust_minmax(DL_artload_raw / norm_deg, 0.0, 1.0)

    # Pocket (dead-end basins) count
    pockets = detect_pockets_via_deadends(grid, min_size=6, connectivity=connectivity)
    # Normalize by map area (coarse)
    h, w, n = grid_shape_size(grid)
    DL_box = robust_minmax(pockets / max(1, n / 32.0), 0.0, 1.0)  # assume ~1 pocket per 32 cells is high

    # Simulation-based term
    EVC = 0.0
    EEC = 0.0
    steps_narrow = 0
    total_steps = 0

    pairs: List[Tuple[Coord, Coord]] = []
    if starts is not None and goals is not None and len(starts) == len(goals) and len(starts) > 0:
        pairs = list(zip(starts, goals))
    else:
        # sample K random distinct pairs
        if nfree >= 2:
            idx = rng.choice(nfree, size=min(K * 2, nfree), replace=False)
            coords = [tuple(map(int, free_coords[i])) for i in idx]
            for i in range(0, len(coords) - 1, 2):
                pairs.append((coords[i], coords[i + 1]))

    if pairs:
        # Build narrow set for steps counting
        narrow_set = set(map(tuple, np.argwhere(narrow_mask)))
        for s, g in pairs:
            path = astar(grid, s, g, connectivity=connectivity, max_steps=max_astar_steps)
            if not path:
                continue
            # For multi-agent simulation, we need multiple paths. Here we approximate by batching pairs in groups of 4.
        # Batch the pairs into groups to simulate simple multi-agent interactions
        B = 4
        for i in range(0, len(pairs), B):
            group = pairs[i:i+B]
            path_list = [astar(grid, s, g, connectivity=connectivity, max_steps=max_astar_steps) for s, g in group]
            # count conflicts and narrow steps
            vc, ec, steps = simulate_conflicts(path_list)
            EVC += vc
            EEC += ec
            total_steps += steps
            for p in path_list:
                steps_narrow += sum(1 for cell in p if cell in narrow_set)

    if total_steps > 0:
        EVC /= total_steps
        EEC /= total_steps
        p_narrow = steps_narrow / total_steps
    else:
        p_narrow = 0.0

    DL_sim_raw = 0.5 * EVC + 0.5 * EEC + 0.5 * p_narrow  # simple equal blend (scaled)
    # Normalize with loose caps
    DL_sim = robust_minmax(DL_sim_raw, 0.0, 0.5)

    DLR = zcombine([DL_corridor, DL_artload, DL_box, DL_sim], weights=[0.25, 0.25, 0.2, 0.3])
    return {
        "DLR": float(DLR),
        "DL_corridor": float(DL_corridor_raw),
        "DL_artload": float(DL_artload_raw),
        "DL_box": float(pockets),
        "DL_sim": float(DL_sim),
        "EVC": float(EVC),
        "EEC": float(EEC),
        "p_narrow": float(p_narrow),
    }


# ------------------ 3.5) Free-space Reachability (FRA) & Detour (FDA) ------------------

def reachability_features(grid: Array2D, connectivity: int = 4) -> Dict[str, float]:
    """
    FRA: Free-space Reachability Area ratio
    = 自由格子图中最大连通分量占全部自由格的比例 ∈ [0,1]。
    直觉：FRA 越小，空间越碎片化、可达性越差。
    """
    G = grid_to_graph(grid, connectivity=connectivity)
    n = G.number_of_nodes()
    if n == 0:
        FRA = 0.0
    else:
        GCC = largest_component(G)
        FRA = float(GCC.number_of_nodes() / n)  # 可达性比例

    # 为了与“难度越大数值越大”的方向一致，构造一个“硬度”项：1 - FRA
    FRA_hard = 1.0 - FRA
    return {
        "FRA": float(FRA),          # 原始可达性
        "FRA_hard": float(FRA_hard) # 难度方向
    }


def detour_features(
    grid: Array2D,
    connectivity: int = 4,
    sample_pairs: int = 300,
    random_state=None,
    detour_cap: float = 3.0,
) -> Dict[str, float]:
    """
    FDA: Free-space Detour Average（归一化）
    做法：
      - 在自由格上随机采样若干对 (u,v)，计算网格最短路径长度 d_grid(u,v)，
        与理想几何距离 d_geo(u,v)（4邻域用曼哈顿，8邻域用切比雪夫）之比 r = d_grid / d_geo。
      - 对 r 取平均得到 avg_ratio（>=1）。再把 (avg_ratio - 1) / (detour_cap - 1) 截断到 [0,1]。
    返回：
      FDA ∈ [0,1]（越大越绕行、越难），以及 avg_ratio 供调参观察。
    """
    rng = _rng_from(random_state)
    fm = free_mask(grid)
    free_coords = np.argwhere(fm == 1)
    if len(free_coords) < 2:
        return {"FDA": 0.0, "FDA_ratio": 1.0}

    # 建图
    G = grid_to_graph(grid, connectivity=connectivity)
    nodes = list(G.nodes())
    if not nodes:
        return {"FDA": 0.0, "FDA_ratio": 1.0}

    # 几何距离函数（与邻接一致）
    def geo_dist(a: Coord, b: Coord) -> int:
        if connectivity == 4:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        else:
            return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    # 采样对
    S = min(sample_pairs, len(nodes))
    idx = rng.choice(len(nodes), size=S, replace=False) if S < len(nodes) else np.arange(len(nodes))
    sampled = [nodes[i] for i in idx]

    # 为加速：对每个 sampled 源做一次单源 BFS，批量取目标
    ratios = []
    for s in sampled:
        lengths = nx.single_source_shortest_path_length(G, s)
        # 随机抽少量目标（含多样性）
        targets_idx = rng.choice(len(nodes), size=min(16, len(nodes)), replace=False)
        for j in targets_idx:
            t = nodes[j]
            if t == s:
                continue
            d_geo = geo_dist(s, t)
            if d_geo == 0:
                continue
            d_grid = lengths.get(t, None)
            if d_grid is None:
                continue
            ratios.append(float(d_grid) / float(d_geo))

    if not ratios:
        return {"FDA": 0.0, "FDA_ratio": 1.0}

    avg_ratio = float(np.mean(ratios))  # >= 1
    # 归一化：把 [1, detour_cap] 映射到 [0,1]
    FDA = (avg_ratio - 1.0) / max(1e-6, (detour_cap - 1.0))
    FDA = float(np.clip(FDA, 0.0, 1.0))
    return {"FDA": FDA, "FDA_ratio": avg_ratio}


# ------------------ Top-level complexity ------------------

DEFAULT_WEIGHTS = {
    # base
    "Size": 0.10,
    "Agents": 0.10,
    "Density": 0.10,
    # new
    "LDD": 0.15,
    "BN": 0.20,
    "MC": 0.15,
    "DLR": 0.20,
    # new (默认先不参与加权；需要时你可以把它们调到非零)
    "FRA": 0.1,  # 注意：组合时我们用的是 1 - FRA（见 compute_complexity 内部处理）
    "FDA": 0.1,
}


def compute_complexity(
    grid: Array2D,
    starts: Optional[List[Coord]] = None,
    goals: Optional[List[Coord]] = None,
    K: int = 30,
    weights: Optional[Dict[str, float]] = None,
    window: Optional[int] = None,
    connectivity: int = 4,
    random_state=None,
    max_astar_steps: int = 10000,
    agents: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute all metrics and return a dict with details and a combined 'Complexity' score.
    """
    h, w, n = grid_shape_size(grid)
    dens = global_density(grid)
    size_base = float(max(h, w))  # representative size for normalization

    # normalize base features to [0,1] with loose ranges
    Size_n = robust_minmax(size_base, 4.0, 128.0)
    Density_n = robust_minmax(dens, 0.0, 0.6)
    Agents_n = 0.0 if agents is None else robust_minmax(float(agents), 1.0, 64.0)

    # sub-features
    ldd = ldd_features(grid, window=window)
    bn = bottleneck_features(grid, connectivity=connectivity)
    mc = connectivity_features(grid, connectivity=connectivity, sample_pairs=200, random_state=random_state)
    dlr = deadlock_features(grid, starts=starts, goals=goals, K=K, connectivity=connectivity,
                            random_state=random_state, max_astar_steps=max_astar_steps)
        # --- 新增 FRA / FDA ---
    fra = reachability_features(grid, connectivity=connectivity)  # {"FRA", "FRA_hard"}
    fda = detour_features(grid, connectivity=connectivity, sample_pairs=300, random_state=random_state)


    # combine
    wts = dict(DEFAULT_WEIGHTS)
    if weights:
        wts.update(weights)

    # Build vector
    feats_01 = {
        "Size": Size_n,
        "Agents": Agents_n,
        "Density": Density_n,
        "LDD": robust_minmax(ldd["LDD"], 0.0, 1.0),
        "BN": robust_minmax(bn["BN"], 0.0, 1.0),
        "MC": robust_minmax(mc["MC"], 0.0, 1.0),
        "DLR": robust_minmax(dlr["DLR"], 0.0, 1.0),
        "FRA": robust_minmax(fra["FRA_hard"], 0.0, 1.0),  # 用难度方向 1 - FRA
        "FDA": robust_minmax(fda["FDA"], 0.0, 1.0),       # 已是 [0,1]

    }

    # Ensure weights sum to 1
    total_w = sum(wts.get(k, 0.0) for k in feats_01.keys())
    if total_w <= 0:
        # fallback equal weights
        eq = 1.0 / len(feats_01)
        wts = {k: eq for k in feats_01.keys()}
    else:
        # normalize
        for k in feats_01.keys():
            wts[k] = wts.get(k, 0.0) / total_w

    Complexity = sum(feats_01[k] * wts[k] for k in feats_01.keys())

    out = {
        # base
        "Size_raw": float(size_base),
        "Agents_raw": float(agents if agents is not None else 0),
        "Density_raw": float(dens),
        # normalized used for combination
        **{f"{k}": float(v) for k, v in feats_01.items()},
        # sub-details
        **ldd,
        **bn,
        **mc,
        **dlr,
        **fra,   # 提供 FRA 和 FRA_hard
        **fda,   # 提供 FDA 与 FDA_ratio

        # final
        "Complexity": float(Complexity),
        "weights_used": {k: float(wts[k]) for k in feats_01.keys()},
    }
    return out


# ------------------ Minimal self-test ------------------

def _toy_map(h=16, w=16, corridors=True) -> Array2D:
    """Create a toy map with a few corridors and obstacles for quick testing."""
    g = np.zeros((h, w), dtype=np.uint8)
    # border
    g[0, :] = 1; g[-1, :] = 1; g[:, 0] = 1; g[:, -1] = 1
    if corridors:
        # vertical wall with two narrow doors
        g[:, w//2] = 1
        g[h//3, w//2] = 0
        g[2*h//3, w//2] = 0
        # extra obstacles to make pockets
        g[h//2, 2: w//2 - 1] = 1
        g[h//2 + 1, 2: w//2 - 1] = 1
    return g


if __name__ == "__main__":
    g = _toy_map()
    res = compute_complexity(g, agents=4, random_state=42)
    # Pretty print a subset
    keys = [
        "Size_raw", "Agents_raw", "Density_raw",
        "LDD", "BN", "MC", "DLR","FDA","FRA",
        "Complexity"
    ]
    print({k: res[k] for k in keys})
