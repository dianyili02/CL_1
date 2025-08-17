# from typing import Optional
# from tqdm import tqdm
# class CurriculumScheduler:
#     def __init__(self, base_map_settings, agent_range: list, episodes_per_stage, stages=None, weights_per_stage=None, threshold=0.8, patience=3, mode: str = "agent"):
#         self.base_map_settings = base_map_settings
#         self.agent_range = agent_range
#         self.episodes_per_stage = episodes_per_stage
#         self.threshold = threshold
#         self.patience = patience
#         self.current_stage = 0
#         self.max_stage = len(agent_range) - 1
#         self.last_stage = 0
#         self.fail_count = 0 
#         self.mode = mode
        
#         self.stage_weights = [
#     {'agents': 1.0, 'size': 1.0, 'density': 1.0, 'bottleneck': 1.0, 'goal_dist': 1.0, 'collision': 1.0},  # Stage 0
#     {'agents': 2.0, 'size': 2.0, 'density': 2.0, 'bottleneck': 2.0, 'goal_dist': 2.0, 'collision': 2.0},  # Stage 1
#     {'agents': 3.0, 'size': 3.0, 'density': 3.0, 'bottleneck': 3.0, 'goal_dist': 3.0, 'collision': 3.0},  # Stage 2
#     ...
# ]


#     def get_current_config(self):
#         return self.stages[self.current_stage]

#     def get_weights_for_stage(self):
#         return self.stage_weights[self.current_stage]

    
#     def update(self, success_rate, pbar=None):
#         if success_rate >= self.threshold:
#             if self.current_stage < self.max_stage:
#                 self.current_stage += 1
#                 self.fail_count = 0
#                 if pbar: pbar.write(f"✅ Progressing to Stage {self.current_stage}")
#         else:
#             self.fail_count += 1
#             if pbar: pbar.write(f"⚠️ Stage {self.current_stage} failed. Fail count: {self.fail_count}")

#     def is_done(self):
#         return self.current_stage >= len(self.stages)


#     def get_updated_map_settings(self):
#         updated_settings = {}
#         for name, config in self.base_map_settings.items():
#             updated = config.copy()
#             updated['num_agents'] = self.agent_range[self.current_stage]
#             updated_settings[name] = updated
#         return updated_settings

#     def step(self, episode):
#         self.current_stage = min(episode // self.episodes_per_stage, self.max_stage)

# # class CurriculumScheduler:
# #     def __init__(self,
# #                  base_map_settings: dict,
# #                  map_range: list,
# #                  episodes_per_stage: int = 100,
# #                  mode: str = "map_size"  # 可选 "agent" 或 "map_size"
# #                  ):
# #         self.base_map_settings = base_map_settings
# #         self.map_range = map_range
# #         self.episodes_per_stage = episodes_per_stage
# #         self.current_stage = 1
# #         self.max_stage = len(map_range)
# #         self.mode = mode

# #     def get_updated_map_settings(self):
# #         updated_settings = {}
# #         for name, args in self.base_map_settings.items():
# #             args = args.copy()
# #             if self.mode == "agent":
# #                 args['num_agents'] = self.map_range[self.current_stage - 1]
# #             elif self.mode == "map_size":
# #                 args['size'] = self.map_range[self.current_stage - 1]
# #             updated_settings[name] = args
# #         return updated_settings

# #     def update(self, success_rate, pbar=None):
# #         if success_rate >= 0.9 and self.current_stage < self.max_stage:
# #             self.current_stage += 1
# #             if pbar:
# #                 pbar.write(f"🎯 晋级！进入 Stage {self.current_stage}")

# #     def step(self, episode):
# #         self.current_stage = min(episode // self.episodes_per_stage, self.max_stage)

# g2rl/curriculum.py
from collections import deque
from copy import deepcopy
from typing import Dict, List, Optional

class CurriculumScheduler:
    """
    每张地图内的 Curriculum 调度：
    - 自动生成 stages（按 agent 数，或可扩展为 map_size/density）
    - 每集结束记录 success，维护滑窗成功率
    - 达标(滑窗SR>=threshold)自动晋级；到达本stage上限但未达标则自动重复本stage
    """
    def __init__(self,
                 base_map_settings: Dict[str, dict],
                 agent_range: List[int],
                 episodes_per_stage: int,
                 stages: Optional[List[Dict[str, dict]]] = None,
                 weights_per_stage: Optional[List[Dict[str, float]]] = None,
                 mode: str = "agent"):
        self.base_map_settings = deepcopy(base_map_settings)
        self.agent_range = list(agent_range)
        self.episodes_per_stage = int(episodes_per_stage)
        self.mode = mode

        # —— 内置阈值与滑窗 ——（可按需调整）
        self.threshold = 0.80    # 达标阈值
        self.window = 50         # 滑窗大小（按 episode）

        # 生成 stages
        self.stages = stages if stages is not None else self._build_stages()
        self.stage_weights = weights_per_stage or self._build_default_weights(len(self.stages))

        self.current_stage = 0
        self.max_stage = len(self.stages) - 1

        # 阶段内部统计
        self._win = deque(maxlen=self.window)  # 最近窗口内的成功(0/1)
        self._ep_in_stage = 0                  # 当前stage内已训练的episode数
        self.best_sr = 0.0

    # ---------- 内部 ----------
    def _build_stages(self) -> List[Dict[str, dict]]:
        stages = []
        for i in range(len(self.agent_range)):
            updated = {}
            for name, cfg in self.base_map_settings.items():
                c = deepcopy(cfg)
                if self.mode == "agent":
                    c["num_agents"] = self.agent_range[i]
                elif self.mode == "map_size":
                    c["size"] = self.agent_range[i]
                elif self.mode == "density":
                    c["density"] = self.agent_range[i]
                updated[name] = c
            stages.append(updated)
        return stages

    @staticmethod
    def _build_default_weights(n: int) -> List[Dict[str, float]]:
        out = []
        for i in range(n):
            w = float(i + 1)
            out.append({
                "agents": w, "size": w, "density": w,
                "bottleneck": w, "goal_dist": w, "collision": w
            })
        return out

    # ---------- 外部接口 ----------
    def get_current_config(self) -> Dict[str, dict]:
        return self.stages[self.current_stage]

    def get_updated_map_settings(self) -> Dict[str, dict]:
        # 向后兼容
        return self.get_current_config()

    def get_weights_for_stage(self) -> Dict[str, float]:
        return self.stage_weights[self.current_stage]

    def add_episode_result(self, success: int):
        """每集结束调用：success ∈ {0,1}"""
        s = int(success)
        self._win.append(s)
        self._ep_in_stage += 1
        if self._win:
            self.best_sr = max(self.best_sr, sum(self._win) / len(self._win))

    def current_window_sr(self) -> float:
        return (sum(self._win) / len(self._win)) if self._win else 0.0

    def ready_to_advance(self) -> bool:
        """窗口满且达标即可晋级"""
        return len(self._win) == self._win.maxlen and self.current_window_sr() >= self.threshold

    def end_of_stage(self) -> bool:
        """到达该 stage 可用 episode 上限"""
        return self._ep_in_stage >= self.episodes_per_stage

    def advance(self, pbar=None):
        if self.current_stage < self.max_stage:
            self.current_stage += 1
            if pbar: pbar.write(f"✅ 进入 Stage {self.current_stage}")
        else:
            if pbar: pbar.write("🏁 已到达最后阶段")

        # 重置阶段统计
        self._win.clear()
        self._ep_in_stage = 0

    def repeat_stage(self, pbar=None):
        """未达标：重复当前 stage（仅重置统计，不改变current_stage）"""
        if pbar: pbar.write(f"🔁 未达标（SR={self.current_window_sr():.2f}<{self.threshold:.2f}），重复当前 Stage {self.current_stage}")
        self._win.clear()
        self._ep_in_stage = 0

    def is_done(self) -> bool:
        """可按需用于早停：到最后阶段且已达标"""
        return self.current_stage >= self.max_stage and self.ready_to_advance()
