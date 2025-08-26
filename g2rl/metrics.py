# import numpy as np


# def manhattan_distance(x_st: int, y_st: int, x_end: int, y_end: int) -> int:
#     return abs(x_end - x_st) + abs(y_end - y_st)
    
    
# def moving_cost(num_steps: int, c_start: list[int], c_goal: list[int]) -> float:
#     return num_steps / (manhattan_distance(*c_start, *c_goal))


# def detour_percentage(num_steps: int, opt_path_len: int) -> float:
#     return (num_steps - opt_path_len) / opt_path_len * 100

import numpy as np
from typing import List


def manhattan_distance(x_st: int, y_st: int, x_end: int, y_end: int) -> int:
    return abs(x_end - x_st) + abs(y_end - y_st)
    
    
# def moving_cost(num_steps: int, c_start: List[int], c_goal: List[int]) -> float:
#     return num_steps / (manhattan_distance(*c_start, *c_goal))

# g2rl/metrics.py
def moving_cost(num_steps: int, c_start, c_goal) -> float:
    """
    返回 num_steps / manhattan(c_start, c_goal)，如果起终点相同，返回 NaN（或返回 1.0 由你定）。
    """
    dx = abs(c_start[0] - c_goal[0])
    dy = abs(c_start[1] - c_goal[1])
    dist = dx + dy
    if dist == 0:
        return float("nan")   # 或者 return 1.0
    return num_steps / float(dist)



# def detour_percentage(num_steps: int, opt_path_len: int) -> float:
#     return (num_steps - opt_path_len) / opt_path_len * 100
def detour_percentage(steps_taken: int, opt_len: int) -> float:
    """
    100 * (steps_taken / opt_len - 1)，当 opt_len <= 0 时返回 NaN
    """
    if opt_len <= 0:
        return float("nan")
    return 100.0 * (steps_taken / float(opt_len) - 1.0)
