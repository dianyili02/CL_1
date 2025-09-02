from typing import Any
from copy import deepcopy
from collections import deque
from typing import Any, Union, List, Tuple, Dict


import torch
import numpy as np

from pogema import pogema_v0, GridConfig
from g2rl.pathfinding import a_star


# ===== A* helper (ensure this exists once) =====
import heapq
import numpy as np
from typing import List, Tuple, Dict

Coord = Tuple[int, int]

def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(start: Coord, goal: Coord, grid: np.ndarray) -> List[Coord]:
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)
    H, W = grid.shape[:2]

    def in_bounds(p: Coord) -> bool:
        return 0 <= p[0] < H and 0 <= p[1] < W

    def passable(p: Coord) -> bool:
        # 0 = free, 1 = obstacle
        return in_bounds(p) and (grid[p[0], p[1]] == 0)

    if start == goal:
        return []
    if not passable(start) or not passable(goal):
        return []

    neighbors = [(-1,0),(1,0),(0,-1),(0,1)]
    open_heap: List[Tuple[int, int, Coord]] = []
    gscore: Dict[Coord, int] = {start: 0}
    came_from: Dict[Coord, Coord] = {}

    heapq.heappush(open_heap, (_manhattan(start, goal), 0, start))
    closed = set()

    while open_heap:
        f, g, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            path: List[Coord] = []
            node = cur
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path  # excludes start, includes goal
        closed.add(cur)
        for dx, dy in neighbors:
            nb = (cur[0]+dx, cur[1]+dy)
            if not passable(nb):
                continue
            tg = g + 1
            if tg < gscore.get(nb, 10**9):
                gscore[nb] = tg
                came_from[nb] = cur
                heapq.heappush(open_heap, (tg + _manhattan(nb, goal), tg, nb))
    return []

# --- Robust import for AnimationMonitor across pogema versions ---
AnimationMonitor = None
try:
    # 新一些版本的写法（如果存在）
    from pogema.animation import AnimationMonitor  # type: ignore
except Exception:
    try:
        # 你当前的 1.3.1 版本：放在 svg_animation 里
        from pogema.svg_animation import AnimationMonitor  # type: ignore
    except Exception:
        try:
            # 某些版本可能直接在顶层暴露
            from pogema import AnimationMonitor  # type: ignore
        except Exception:
            AnimationMonitor = None  # 动画不可用，后续判断关闭



class Grid:
    '''Basic grid container'''
    def __init__(self, obstacles: np.ndarray):
        assert obstacles.ndim == 2 and obstacles.shape[0] == obstacles.shape[1]
        self.obstacles = obstacles.copy().astype(bool)
        self.size = obstacles.shape[0]

    # def is_obstacle(self, h: int, w: int) -> Union[bool, None]:
    #     if 0 <= h <= self.size and 0 <= w <= self.size:
    #         return self.obstacles[h, w]
    #     else:
    #         return False
    def is_obstacle(self, h: int, w: int) -> Union[bool, None]:
        if 0 <= h < self.size and 0 <= w < self.size:
            return self.obstacles[h, w]
        else:
            return False



class G2RLEnv:
    '''Environment for MAPF G2RL implementation'''
    def __init__(
            self,
            size: int = 50,
            num_agents: int = 3,
            density: Union[float, None] = None,
            map: Union[str, List, None] = None,
            obs_radius: int = 7,
            cache_size: int = 4,
            r1: float = -0.01,
            r2: float = -0.1,
            r3: float = 0.1,
            seed: int = 42,
            animation: bool = True,
            collission_system: str = 'soft',
            on_target: str = 'restart',
            max_episode_steps: int = 64,
        ):
        self.time_idx = 1
        self.num_agents = num_agents
        self.obs_radius = obs_radius
        self.cache_size = cache_size
        self.r1, self.r2, self.r3 = r1, r2, r3
        self.collission_system = collission_system
        self.on_target = on_target
        self.obs, self.info = None, None

        self._set_env(
            map,
            seed=seed,
            size=size,
            density=density,
            max_episode_steps=max_episode_steps,
            animation=animation)

        # self.actions = [
        #     ('idle', 0, 0),
        #     ('up', -1, 0),
        #     ('down', 1, 0),
        #     ('left', 0, -1),
        #     ('right', 0, 1),
        # ]
        # 测试用
        # g2rl/environment.py 里（示例）
        self._actions = {
            0: (0, 0),   # stay
            1: (-1, 0),  # up (y-1)
            2: (1, 0),   # down (y+1)   # ←← 修复：原来是 (0,0) 错了
            3: (0, -1),  # left (x-1)
            4: (0, 1),   # right (x+1)
        }
        self._init_actions() 


    def _init_actions(self):
        """
        确保环境里有动作表和映射。默认 5 个动作：idle/up/down/left/right
        """
        if not hasattr(self, "actions") or self.actions is None:
            self.actions = ['idle', 'up', 'down', 'left', 'right']
        # 可选：如果你本来就定义了别的动作集，这里别覆盖
        self._action_to_delta = {
            'idle':  (0, 0),
            'up':    (-1, 0),
            'down':  (1, 0),
            'left':  (0, -1),
            'right': (0, 1),
        }
        self._n_actions = len(self.actions)

    def _get_reward(self, case: int, N: int = 0) -> float:
        rewards = [self.r1, self.r1 + self.r2, self.r1 + N * self.r3]
        return rewards[case]

    def _set_env(
            self,
            map: Union[str, List, None],
            size: int = 48,
            density: float = 0.392,
            seed: int = 42,
            max_episode_steps: int = 64,
            animation: bool = True,
        ):
        if map is not None:
            self.grid_config = GridConfig(
                map=map,
                seed=seed,
                observation_type='MAPF',
                on_target=self.on_target,
                num_agents=self.num_agents,
                obs_radius=self.obs_radius,
                collission_system=self.collission_system,
                max_episode_steps=max_episode_steps,
            )
            self.size = self.grid_config.size
        else:
            self.grid_config = GridConfig(
                size=size,
                density=density,
                seed=seed,
                observation_type='MAPF',
                on_target=self.on_target,
                num_agents=self.num_agents,
                obs_radius=self.obs_radius,
                collission_system=self.collission_system,
                max_episode_steps=max_episode_steps,
            )
            self.size = size

        self.env = pogema_v0(grid_config=self.grid_config)
        if animation:
            self.env = AnimationMonitor(self.env)

    # def _set_global_guidance(self, obs: List[Dict]):
    #     # grid = Grid(obs[0]['global_obstacles'])
    #     # coords = [[ob['global_xy'], ob['global_target_xy']] for ob in obs]
    #     # self.global_guidance = [a_star(st, tg, grid) for st, tg in coords]
    #     coords = []
    #     for ob in obs:
    #         start = tuple(ob['global_xy'])
    #         goal = tuple(ob['global_target_xy'])
    #         coords.append((start, goal))

    # # ✅ 提取 obstacle 网格
    #     grid_array = self.env.grid.get_obstacles().tolist()

    # # ✅ 使用 numpy 数组传给 a_star
    #     # print(f"[DEBUG] grid_array type: {type(grid_array)}")           # 应该是 <class 'list'>
    #     # print(f"[DEBUG] grid_array[0] type: {type(grid_array[0])}")     # 应该是 <class 'list'>
    #     # print(f"[DEBUG] grid_array shape: {len(grid_array)}x{len(grid_array[0])}")

    #     opt_path = [tuple(state['global_xy'])] + list(env.global_guidance[target_idx])


    # def _set_global_guidance(self, obs):
    # # """
    # # Build per-agent global guidance using A*.
    # # - obs: list/dict with keys 'global_xy' and 'global_target_xy' for each agent
    # # - self.grid / self._grid: binary grid (0 free, 1 obstacle)
    # # Result:
    # #   self.global_guidance: List[List[Tuple[int,int]]] for each agent
    # #   Each path excludes start and includes goal.
    # # """
    # # 取出二值栅格
    #     if hasattr(self, "grid") and self.grid is not None:
    #         grid_array = np.array(self.grid, dtype=int)
    #     elif hasattr(self, "_grid") and self._grid is not None:
    #         grid_array = np.array(self._grid, dtype=int)
    #     else:
    #         raise RuntimeError("No grid array found on environment (expected self.grid or self._grid).")

    # # 组装 (start, goal)
    #     coords = []
    #     n_agents = getattr(self, "num_agents", None)
    #     if n_agents is None:
    #     # 兜底从 obs 推断
    #         n_agents = len(obs) if hasattr(obs, "__len__") else 1

    #     for i in range(n_agents):
    #         item = obs[i]
    #     # 支持两种可能字段名
    #         s = item.get('global_xy', item.get('xy', None))
    #         g = item.get('global_target_xy', item.get('target_xy', None))
    #         if s is None or g is None:
    #             raise KeyError(f"obs[{i}] missing 'global_xy'/'global_target_xy' (or 'xy'/'target_xy'). Got keys: {list(item.keys())}")
    #         s = tuple(map(int, s))
    #         g = tuple(map(int, g))
    #         coords.append((s, g))

    # # 为每个 agent 计算路径
    #     guidance = []
    #     for st, tg in coords:
    #         path = a_star(st, tg, grid_array)  # [] if unreachable
    #         guidance.append(path)

    #     self.global_guidance = guidance
    
    def _set_global_guidance(self, obs):
    # """
    # 使用 pogema 的网格对象构建每个智能体的全局指引（A* 路径）。
    # - obs: reset() 返回的观测列表，含 'global_xy' 与 'global_target_xy'
    # - self.env.grid: pogema 的网格对象，get_obstacles() -> 0/1 numpy 数组（0=free,1=obstacle）
    # 结果：
    #   self.global_guidance = List[List[Tuple[int,int]]]
    #   每条路径：不含起点，含终点；若不可达则为空列表。
    # """
    # 1) 从 pogema 取二值栅格
        if hasattr(self, "env") and hasattr(self.env, "grid") and self.env.grid is not None:
        # get_obstacles() 可能返回 numpy 数组或 list，这里统一成 np.ndarray[int]
            grid_raw = self.env.grid.get_obstacles()
            grid_array = np.array(grid_raw, dtype=int)
        else:
            raise RuntimeError("pogema env.grid 不存在或未初始化，无法生成全局路径。请确保先调用 self.env.reset()。")

    # 2) 组装 (start, goal)
        coords = []
        n_agents = getattr(self, "num_agents", len(obs) if hasattr(obs, "__len__") else 1)
        for i in range(n_agents):
            item = obs[i]
        # 兼容两种字段名
            s = item.get('global_xy', item.get('xy', None))
            g = item.get('global_target_xy', item.get('target_xy', None))
            if s is None or g is None:
                raise KeyError(
                    f"obs[{i}] 缺少 'global_xy'/'global_target_xy' (或 'xy'/'target_xy')。现有键: {list(item.keys())}"
                )
            s = tuple(map(int, s))
            g = tuple(map(int, g))
            coords.append((s, g))

    # 3) 为每个 agent 计算 A* 路径
        guidance = []
        for st, tg in coords:
            path = a_star(st, tg, grid_array)  # 不可达则返回 []
            guidance.append(path)

        self.global_guidance = guidance









    def save_animation(self, path):
        self.env.save_animation(path)

    def get_action_space(self) -> List[int]:
        self._init_actions() 
        return list(range(len(self.actions)))

    def reset(self) -> Tuple[List, List]:
        # print("[DEBUG] reset 被调用，是否设置 goals")
        self.time_idx = 1
        self.obs, self.info = self.env.reset()
        self._set_global_guidance(self.obs)
        # print("[DEBUG] reset() 执行中，goals 被设置了！")
        self.goals = [ob['global_target_xy'] for ob in self.obs] 
        self.view_cache = []
        for i, (ob, guidance) in enumerate(zip(self.obs, self.global_guidance)):
            # guidance.remove(ob['global_xy'])
            view = self._get_local_view(ob, guidance)
            view_cache = [np.zeros_like(view) for _ in range(self.cache_size - 1)] + [view]
            self.view_cache.append(deque(view_cache, self.cache_size))
            self.obs[i]['view_cache'] = np.array(self.view_cache[-1])
            self._init_actions() 
        # print(f"[env.reset()] Goals 已设置：{self.goals}")
        return self.obs, self.info

    def _reset_agent(self, i: int, ob: Dict[str, Any]) -> Dict[str, Any]:
        grid = Grid(ob['global_obstacles'])
        # self.global_guidance[i] = a_star(ob['global_xy'], ob['global_target_xy'], grid)
        grid_array = self.env.grid.get_obstacles().tolist()
        self.global_guidance[i] = a_star(ob['global_xy'], ob['global_target_xy'], grid_array)

        # self.global_guidance[i].remove(ob['global_xy'])
        cur = (int(ob['global_xy'][0]), int(ob['global_xy'][1]))
        g = self.global_guidance[i]
        if g and tuple(g[0]) == cur:  # 仅当首元素就是当前位置时前进一步
            g.pop(0)


        view = self._get_local_view(ob, self.global_guidance[i])
        view_cache = [np.zeros_like(view) for _ in range(self.cache_size - 1)] + [view]
        self.view_cache[i] = deque(view_cache, self.cache_size)
        ob['view_cache'] = np.array(self.view_cache[i])
        return ob

    def _get_local_view(
            self,
            obs: Dict[str, Any],
            global_guidance: np.ndarray,
        ) -> np.ndarray:
        local_coord = self.obs_radius

        local_guidance = np.zeros_like(obs['agents'])
        local_size = local_guidance.shape[0]
        delta = [global_coord - local_coord for global_coord in obs['global_xy']]
        for global_cell in global_guidance:
            h = global_cell[0] - delta[0]
            w = global_cell[1] - delta[1]
            if 0 <= h < local_size and 0 <= w < local_size:
                local_guidance[h, w] = 1

        curr_agent = np.zeros_like(obs['agents'])
        curr_agent[local_coord, local_coord] = 1
        return np.dstack(
            (
                curr_agent,
                obs['obstacles'],
                obs['agents'],
                local_guidance,
            )
        )

    def step(self, actions: List[int]) -> Tuple[List, ...]:
        self._init_actions() 
        # 统一为动作名列表
        norm_actions = []
        for a in actions:
            if isinstance(a, int):
                norm_actions.append(self.actions[a % self._n_actions])
            else:
            # 假定已经是字符串动作
                norm_actions.append(str(a))
                
        conflict_points = set()
        obs, reward, terminated, truncated, info = self.env.step(actions)
        # calculate reward
        for i, (action, ob, status) in enumerate(zip(actions, obs, info)):
            if status['is_active']:
                new_point = ob['global_xy']
                # conflict
                if self.actions[action] != 'idle' and new_point == self.obs[i]['global_xy']:
                    reward[i] = self._get_reward(1)
                    if self.collission_system != 'block_both':
                        # another agent (block strategy is considered)
                        if ob['global_obstacles'][new_point] == 0:
                            conflict_points.add(new_point)
                # global guidance cell
                elif new_point in self.global_guidance[i]:
                    new_point_idx = self.global_guidance[i].index(new_point)
                    reward[i] = self._get_reward(2, new_point_idx + 1)
                    # update global guidance
                    if self.on_target == 'nothing':
                        if new_point == self.global_guidance[i][-1]:
                            self.global_guidance[i] = self.global_guidance[i][-1:]
                        else:
                            self.global_guidance[i] = self.global_guidance[i][new_point_idx + 1:]
                    else:
                        self.global_guidance[i] = self.global_guidance[i][new_point_idx + 1:]
                        if len(self.global_guidance[i]) == 0:
                            ob = self._reset_agent(i, ob)
                # free cell
                else:
                    reward[i] = self._get_reward(0)

                # update history of observations
                view = self._get_local_view(ob, self.global_guidance[i])
                self.view_cache[i].append(view)

            obs[i]['view_cache'] = np.array(self.view_cache[i])

        # recalculate reward if strategy is not blocking
        if self.collission_system != 'block_both':
            for i, (ob, status) in enumerate(zip(obs, info)):
                if status['is_active'] and ob['global_xy'] in conflict_points:
                    reward[i] = self._get_reward(1)

        self.obs, self.info = obs, info
        self.time_idx += 1

        return obs, reward, terminated, truncated, info

    @staticmethod
    def select_map_env(map_settings, default='regular'):
    # """
    # 从用户输入选择地图类型，并返回对应 G2RLEnv 实例。

    # 参数：
    #     map_settings (dict): 预定义地图配置，如 {'regular': {...}, ...}
    #     default (str): 输入无效时默认地图类型

    # 返回：
    #     env (G2RLEnv): 构造好的环境实例
    #     map_type (str): 选用的地图类型名称
    # """
        map_keys = list(map_settings.keys())
    
        print("\n🗺️ 可选地图类型：")
        for i, key in enumerate(map_keys):
            print(f"  [{i}] {key} -> {map_settings[key]}")

        choice = input("请输入地图类型（名称或编号）: ").strip().lower()

        if choice in map_settings:
            map_type = choice
        else:
            try:
                index = int(choice)
                map_type = map_keys[index]
            except:
                print(f"⚠️ 输入无效，默认使用 {default}")
                map_type = default

        cfg = map_settings[map_type]
        env = G2RLEnv(size=cfg['size'], num_agents=cfg['num_agents'], density=cfg['density'])

        print(f"✅ 当前使用地图：{map_type}, 配置：{cfg}")
        return env, map_type