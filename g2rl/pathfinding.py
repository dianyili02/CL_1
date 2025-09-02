# g2rl/pathfinding.py
import numpy as np
from heapq import heappop, heappush
from typing import Tuple, Optional, Iterable, Any

Coord = Tuple[int, int]
INF = 10**9

# ---------- A* 所需：GridMemory（把局部obstacles贴到全局记忆） ----------
class GridMemory:
    def __init__(self, start_r: int = 64):
        self._memory = np.zeros((start_r * 2 + 1, start_r * 2 + 1), dtype=np.bool_)

    @staticmethod
    def _try_to_insert(x, y, source, target) -> bool:
        r = source.shape[0] // 2
        try:
            target[x - r:x + r + 1, y - r:y + r + 1] = source
            return True
        except ValueError:
            return False

    def _increase_memory(self):
        m = self._memory
        r = self._memory.shape[0]
        self._memory = np.zeros((r * 2 + 1, r * 2 + 1), dtype=np.bool_)
        assert self._try_to_insert(r, r, m, self._memory)

    def update(self, x: int, y: int, obstacles: np.ndarray):
        # obstacles 是以 (x,y) 为中心的局部补丁（0=free, 1=obs），pogema 默认就是这样
        while True:
            r = self._memory.shape[0] // 2
            if self._try_to_insert(r + x, r + y, obstacles, self._memory):
                break
            self._increase_memory()

    def is_obstacle(self, x: int, y: int) -> bool:
        r = self._memory.shape[0] // 2
        if -r <= x <= r and -r <= y <= r:
            return bool(self._memory[r + x, r + y])
        return False  # 记忆外一律视为未知->可通行（保守起见你也可改成 True）


# ---------- 0/1 网格适配器（让 a_star 既能吃 GridMemory 也能吃数组） ----------
class _GridArrayAdapter:
    def __init__(self, grid_any: Any):
        if hasattr(grid_any, "shape"):
            self.grid = grid_any
            self.H, self.W = grid_any.shape
            self._get = lambda x, y: int(self.grid[x, y])
        else:
            self.grid = grid_any
            self.H, self.W = len(grid_any), len(grid_any[0])
            self._get = lambda x, y: int(self.grid[x][y])

    def is_obstacle(self, x, y) -> bool:
        return not (0 <= x < self.H and 0 <= y < self.W) or self._get(x, y) != 0


# ---------- A* 主体（支持两种参数顺序） ----------
class Node:
    def __init__(self, coord: Coord = (INF, INF), g: int = 0, h: int = 0):
        self.i, self.j = coord
        self.g = g
        self.h = h
        self.f = g + h
    def __lt__(self, other):
        if self.f != other.f: return self.f < other.f
        if self.g != other.g: return self.g < other.g
        if self.i != other.i: return self.i < other.i
        return self.j < other.j

def _heuristic(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _coerce_args(a, b, c):
    is_xy = lambda v: isinstance(v, (tuple, list)) and len(v) == 2
    if is_xy(a) and is_xy(b):
        start, goal, grid = (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), c
    else:
        grid, start, goal = a, (int(b[0]), int(b[1])), (int(c[0]), int(c[1]))
    grid_iface = grid if hasattr(grid, "is_obstacle") else _GridArrayAdapter(grid)
    return grid_iface, start, goal

def a_star(arg1, arg2, arg3, max_steps: int = 100000) -> Optional[Iterable[Coord]]:
    """
    A* on 0/1 grid or GridMemory (4-neigh, 0=free, 1=obstacle)
    Accepts: a_star(grid, start, goal)  or  a_star(start, goal, grid)
    Returns path [start,...,goal] or None
    """
    grid, start, goal = _coerce_args(arg1, arg2, arg3)
    if grid.is_obstacle(*start) or grid.is_obstacle(*goal):
        return None

    open_q, closed = [], {start: None}
    heappush(open_q, Node(start, 0, _heuristic(start, goal)))
    steps = 0

    while open_q and steps < max_steps:
        u = heappop(open_q); steps += 1
        if (u.i, u.j) == goal:
            path, cur = [], (u.i, u.j)
            while cur is not None:
                path.append(cur); cur = closed[cur]
            return list(reversed(path))
        for n in ((u.i-1,u.j), (u.i+1,u.j), (u.i,u.j-1), (u.i,u.j+1)):
            if not grid.is_obstacle(*n) and n not in closed:
                heappush(open_q, Node(n, u.g + 1, _heuristic(n, goal)))
                closed[n] = (u.i, u.j)
    return None


# ---------- A* 队友代理（供 train0829.py 里 teammates 使用） ----------
class AStarAgent:
    """
    兼容 Pogema 风格 observation：
      obs['xy'] -> (x,y)
      obs['target_xy'] -> (tx,ty)
      obs['obstacles'] -> 局部障碍补丁（0/1），以 xy 为中心
    """
    def __init__(self, grid: Optional[GridMemory] = None, seed: int = 0):
        # 动作集合（默认：0=原地, 上, 下, 左, 右）；若环境不同可自行调整
        try:
            from pogema import GridConfig
            moves = GridConfig().MOVES
        except Exception:
            moves = [(0,0), (-1,0), (1,0), (0,-1), (0,1)]
        self._moves = moves
        self._reverse_actions = {tuple(self._moves[i]): i for i in range(len(self._moves))}
        self._gm = grid if isinstance(grid, GridMemory) else GridMemory()
        self._saved_xy: Optional[Coord] = None
        self._rnd = np.random.default_rng(seed)

    def clear_state(self):
        self._saved_xy = None
        self._gm = GridMemory()

    def act(self, obs: dict) -> int:
        xy = tuple(map(int, obs.get('xy') or obs.get('global_xy')))
        tgt = tuple(map(int, obs.get('target_xy') or obs.get('global_target_xy')))
        obstacles = obs['obstacles']  # Pogema 提供的局部补丁（0/1）

        # 连续步一致性（防止外部一次跳多格）
        if self._saved_xy is not None:
            dx = abs(self._saved_xy[0] - xy[0]) + abs(self._saved_xy[1] - xy[1])
            if dx > 1:
                # 新 episode 时请调用 clear_state()
                self.clear_state()

        # 更新记忆并寻路
        self._gm.update(*xy, obstacles)
        path = a_star(xy, tgt, self._gm)

        # 目标附近或无路 -> 原地/随机
        if not path or len(path) <= 1:
            action = 0  # 原地
        else:
            (x, y), (nx, ny), *_ = path
            step = (nx - x, ny - y)
            action = self._reverse_actions.get(step, 0)

        self._saved_xy = xy
        return action


class BatchAStarAgent:
    def __init__(self):
        self.astar_agents = {}

    def act(self, observations):
        actions = []
        for idx, obs in enumerate(observations):
            if idx not in self.astar_agents:
                self.astar_agents[idx] = AStarAgent()
            actions.append(self.astar_agents[idx].act(obs))
        return actions

    def reset_states(self):
        self.astar_agents.clear()
