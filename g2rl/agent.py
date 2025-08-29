import random
import copy
from typing import Any
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from g2rl.network import CRNNModel
from g2rl.environment import G2RLEnv
from g2rl.utils import PrioritizedReplayBuffer
from typing import Any, List, Dict


class G2RLAgent:
    '''Inference implementation of G2RL agent'''
    def __init__(
            self,
            model: torch.nn.Module,
            action_space: List[int],
            epsilon: float = 0.1,
            device: str = 'cpu',
            lifelong: bool = True,
        ):
        self.device = device
        self.epsilon = epsilon
        self.action_space = action_space
        self.q_network = model.to(self.device)
        self.q_network.eval()
        self.lifelong = lifelong

    def act(self, state: Dict[str, Any]) -> int:
        state = state['view_cache']
        # check not lifelong status
        local_guidance = state[-1,:,:,-1]
        agent_coord = local_guidance.shape[0] // 2
        if not self.lifelong and \
            local_guidance[agent_coord,agent_coord] == 1 == local_guidance.sum():
            return 0
        # lifelong strategy
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        if random.random() <= self.epsilon:
            return random.choice(self.action_space)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()


class DDQNAgent:
    '''Implementation of DDQN agent with a prioritized sumtree reply buffer'''
    def __init__(
            self,
            q_network,
            model: torch.nn.Module,
            action_space: List[int],
            gamma: float = 0.95,
            tau: float = 0.01,
            initial_epsilon: float = 1.0,
            final_epsilon: float = 0.1,
            decay_range: int = 5_000,
            lr: float = 0.001,
            replay_buffer_size: int = 1000,
            device: str = 'cpu',
            alpha: float = 0.6,
            beta: float = 0.4,
            # ↓↓↓ 新增两个可选参数 ↓↓↓
            use_channels=(1, 0, 3, 2),  # 通道重排：障碍, 自车, 目标, 其他
            action_map=None,
            obs_preprocessor=None):           # 动作索引映射（网络->环境）

        self.device = device
        self.action_space = action_space
        self.replay_buffer = PrioritizedReplayBuffer(replay_buffer_size, alpha)
        self.q_network = q_network


        self.obs_preprocessor = obs_preprocessor
        self.tau = tau
        # self.q_network = model
        self.target_network = copy.deepcopy(model)
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        self.target_network.eval()
        
        self.gamma = gamma
        self.final_epsilon = final_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay = (initial_epsilon - final_epsilon) / decay_range
        self.optimizer = Adam(self.q_network.parameters(), lr=lr)
        self.beta = beta
         # 保存设置
        self.use_channels = tuple(use_channels)
        # 默认恒等映射 [0,1,2,3,4]；如你之前求得[0,2,1,3,4]就传进来
        self.action_map = list(action_map) if action_map is not None else list(range(len(self.action_space)))

    def save_weights(self, path: str):
        torch.save(self.target_network.state_dict(), path)


#     def _to_batch_tensor(self, s):
#         if self.obs_preprocessor is None:
#             raise RuntimeError("obs_preprocessor 未设置：请在创建 Agent 后赋值或在 __init__ 传入。")
#         x = self.obs_preprocessor(s)
#         if not torch.is_tensor(x):
#             x = torch.tensor(x, dtype=torch.float32, device=self.device)
#         else:
#             x = x.to(self.device, dtype=torch.float32)
#         if x.dim() == 4:
#             x = x.unsqueeze(0)
#         elif x.dim() != 5:
#             raise RuntimeError(f"预处理后的观测维度非法：dim={x.dim()}，期望 4 或 5")
#         return x  # [1,C,D,H,W]


#     def store(self, state: Dict[str, Any], action: int, reward: float, next_state: Dict[str, Any], terminated: bool):
#         state_cache = state['view_cache']
#         next_state_cache = next_state['view_cache']
#         transition = (state_cache, action, reward, next_state_cache, terminated)
#         # 统一预处理 state / next_state
#         state_tensor = self._to_batch_tensor(state)        # [1,C,D,H,W]
#         next_tensor  = self._to_batch_tensor(next_state)   # [1,C,D,H,W]

#         with torch.no_grad():
#     # Q(s,a)
#             q_all = self.q_network(state_tensor)           # [1, num_actions]
#             curr_Q = q_all[0, action]

#     # max_a' Q_target(s', a')
#             q_next_all = self.target_network(next_tensor)  # [1, num_actions]
#             max_next_Q = q_next_all.max(dim=1)[0].item()

# # TD 误差用于优先级（按你原来的 formula）
#         target_Q = reward + (0.0 if done else self.gamma * max_next_Q)
#         td_error = float(abs(target_Q - curr_Q.item()))

# # 入库时，建议存“预处理好的张量去掉 batch 维”，方便后面堆叠：
#         state_proc = state_tensor.squeeze(0)      # [C,D,H,W]
#         next_proc  = next_tensor.squeeze(0)       # [C,D,H,W]

#         self.replay_buffer.add(state_proc, action, reward, next_proc, done, priority=td_error)


#     def align_target_model(self):
#         for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
#             target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

#     # def act(self, state: Dict[str, Any]) -> int:
#     #     state = state['view_cache']
#     #     state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
#     #     if random.random() <= self.epsilon:
#     #         return random.choice(self.action_space)
#     #     with torch.no_grad():
#     #         q_values = self.q_network(state)
#     #     return torch.argmax(q_values).item()
#     # 测试用
#     def act(self, state):
#         # 1) 从 obs 里拿时序视野并重排通道
#         V = state['view_cache']  # 形状 [T,H,W,C]
#         # 安全转换 + 重排到训练时顺序（我们探针推断为 [障碍, 自车, 目标, 其他] = (1,0,3,2)）
#         V = np.asarray(V, dtype=np.float32)[..., list(self.use_channels)]  # [T,H,W,C']
#         # 转成 [1,C,T,H,W]
#         x = torch.from_numpy(V).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)

#         # 2) 前向 & 选网络动作
#         with torch.no_grad():
#             q = self.q_network(x)
#             net_action = int(torch.argmax(q, dim=1).item())

#         # 3) 网络动作 -> 环境动作（如你之前校准出的 [0,2,1,3,4]）
#         env_action = self.action_map[net_action]
#         return env_action

    

#     def retrain(self, batch_size: int) -> float:
#         if len(self.replay_buffer) < batch_size:
#             return
        
#         samples, indices, weights = self.replay_buffer.sample(batch_size, self.beta)
#         states, actions, rewards, next_states, dones = zip(*samples)
        
#         # 1) 堆叠成张量
#         states      = torch.stack(states_list, dim=0).to(self.device)        # [B,C,D,H,W]
#         next_states = torch.stack(next_states_list, dim=0).to(self.device)   # [B,C,D,H,W]
#         actions     = torch.as_tensor(actions_list, dtype=torch.long, device=self.device)   # [B]
#         rewards     = torch.as_tensor(rewards_list, dtype=torch.float32, device=self.device) # [B]
#         dones       = torch.as_tensor(dones_list, dtype=torch.bool, device=self.device)      # [B]

# # 2) 前向
#         q_curr_all = self.q_network(states)            # [B, num_actions]
#         q_next_all = self.target_network(next_states)  # [B, num_actions]

#         curr_Q     = q_curr_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)     # [B]
#         max_next_Q = q_next_all.max(dim=1)[0]                                     # [B]

# # 3) 目标
#         target_Q = rewards + (~dones).float() * self.gamma * max_next_Q           # [B]

# # 4) 损失 & 反传（保留你原本的 PER 权重加权逻辑）
#         loss = F.smooth_l1_loss(curr_Q, target_Q.detach(), reduction="none")
        
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
        
#         self.align_target_model()
        
#         errors = torch.abs(curr_Q - target_Q).detach().cpu().numpy()
#         self.replay_buffer.update_priorities(indices, errors)
        
#         if self.epsilon > self.final_epsilon:
#             self.epsilon -= self.epsilon_decay
            
#         return loss.item()
    # ===== 放在 DDQNAgent 类里：工具，把单条观测 -> [1,C,D,H,W] =====
    def _to_batch_tensor(self, s):

        import torch, numpy as np

        if self.obs_preprocessor is not None:
            x = self.obs_preprocessor(s)
            if not torch.is_tensor(x):
                x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            else:
                x = x.to(self.device, dtype=torch.float32)
            if x.dim() == 4:
                x = x.unsqueeze(0)
            elif x.dim() != 5:
                raise RuntimeError(f"obs_preprocessor 输出维度非法: dim={x.dim()}")
            return x  # [1,C,D,H,W]

    # 没传 obs_preprocessor，则走 view_cache + 通道重排
        V = s['view_cache']  # 期望 [T,H,W,C]
        import numpy as np
        V = np.asarray(V, dtype=np.float32)[..., list(self.use_channels)]  # [T,H,W,C']
        x = torch.from_numpy(V).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)  # [1,C,T,H,W]
        return x

# ===== 覆盖 store：统一预处理，修复 done 变量，写入 PER 优先级 =====
    def store(self, state, action: int, reward: float, next_state, terminated: bool):
        import torch
    # 预处理为 batch
        state_b = self._to_batch_tensor(state)        # [1,C,D,H,W]
        next_b  = self._to_batch_tensor(next_state)   # [1,C,D,H,W]

        with torch.no_grad():
            q_now   = self.q_network(state_b)         # [1,A]
            curr_Q  = q_now[0, int(action)]
            q_next  = self.target_network(next_b)     # [1,A]
            maxQn   = q_next.max(dim=1)[0].item()

    # 用传进来的 terminated，别再用未定义的 done
        target_Q = float(reward) + (0.0 if bool(terminated) else self.gamma * maxQn)
        td_error = abs(target_Q - curr_Q.item())

    # 入库时存去掉 batch 的 [C,D,H,W]，便于后续 stack 成 [B,C,D,H,W]
        state_proc = state_b.squeeze(0).detach()   # [C,D,H,W]
        next_proc  = next_b.squeeze(0).detach()    # [C,D,H,W]

    # PER：add(..., priority=td_error)
        # --- 兼容多种 add(...) 签名 ---
        transition = (state_proc, int(action), float(reward), next_proc, bool(terminated))
        added = False
# 1) 有的实现: add(s, a, r, s', d, priority)
        try:
            self.replay_buffer.add(transition, td_error)
            added = True
        except TypeError:
            pass

# 2) 有的实现: add((s, a, r, s', d), priority)
        if not added:
            try:
                self.replay_buffer.add(td_error, transition)
                added = True
            except TypeError:
                pass

# 3) 简单 buffer: add(s, a, r, s', d)
        if not added:
            try:
                self.replay_buffer.add(transition)
                added = True
            except TypeError:
                pass

        # 4) 有些实现用 push(...)
        if not added and hasattr(self.replay_buffer, "push"):
            try:
        # 优先 push(transition, priority)
                self.replay_buffer.push(transition, td_error)
                added = True
            except TypeError:
                try:
                    self.replay_buffer.push(transition)
                    added = True
                except TypeError:
                    pass

        if not added:
    # 打印签名，便于你确认具体需要哪种调用
            import inspect
            try:
                sig = inspect.signature(self.replay_buffer.add)
            except Exception:
                sig = "unknown"
            raise RuntimeError(f"Unsupported replay_buffer.add signature: {sig}. "
                       f"Tried (transition, priority)/(priority, transition)/(transition)/push(...)")

    @torch.no_grad()
    def align_target_model(self):
    # 兜底：若没建过 target，就先硬拷一次
        if not hasattr(self, "target_network") or self.target_network is None:
            import copy
            self.target_network = copy.deepcopy(self.q_network).to(self.device)
            self.target_network.eval()

        tau = float(getattr(self, "tau", 0.01))
        tau = max(0.0, min(1.0, tau))  # clamp 到 [0,1]

        for tgt, src in zip(self.target_network.parameters(), self.q_network.parameters()):
        # θ_tgt ← τ·θ_src + (1-τ)·θ_tgt
            tgt.data.mul_(1.0 - tau).add_(tau * src.data)



# ===== 覆盖 act（保留你原来的通道重排 + action_map；可按需加 epsilon 探索）=====
    def act(self, state):
        import torch, numpy as np, random
    # ε-贪心（如不需要探索可去掉这一段）
        if hasattr(self, "epsilon") and random.random() <= float(self.epsilon):
            if self.action_map is None:
                return random.choice(self.action_space)
            else:
            # 网络动作空间 -> 环境动作空间
                net_a = random.choice(range(len(self.action_map)))
                return int(self.action_map[net_a])

    # 前向选择
        x = self._to_batch_tensor(state)  # [1,C,D,H,W]
        with torch.no_grad():
            q = self.q_network(x)
            net_action = int(torch.argmax(q, dim=1).item())

    # 网络动作 -> 环境动作
        if self.action_map is None:
            return net_action
        return int(self.action_map[net_action])

# ===== 覆盖 retrain：稳健解包、堆叠 batch、前向、加权损失、反传、更新 PER =====
    def retrain(self, batch_size: int) -> float:
        import torch
        import torch.nn.functional as F

        if len(self.replay_buffer) < batch_size:
            return 0.0

    # 兼容返回格式： (samples, indices, weights) 或 dict
        sampled = self.replay_buffer.sample(batch_size, getattr(self, "beta", 0.4))

    # 标准 PER 返回 (samples, indices, weights)
        if isinstance(sampled, (tuple, list)) and len(sampled) == 3:
            samples, indices, weights = sampled
        elif isinstance(sampled, dict):
        # 字典风格
            samples  = sampled.get("samples")
            indices  = sampled.get("indices")
            weights  = sampled.get("weights")
            if samples is None:
            # 直接字典里就有各字段
                states      = sampled.get("states") or sampled.get("obs")
                next_states = sampled.get("next_states") or sampled.get("next_obs")
                actions     = sampled["actions"]
                rewards     = sampled["rewards"]
                dones       = sampled.get("dones") or sampled.get("terminated") or sampled.get("done")
                weights     = sampled.get("weights")
            # 统一成 list[tuple]
                samples = list(zip(states, actions, rewards, next_states, dones))
                indices = sampled.get("indices")
        else:
        # 只返回样本列表
            samples, indices, weights = sampled, None, None

    # 解包五元组
        states, actions, rewards, next_states, dones = zip(*samples)

    # 堆叠为张量
        def _to_tensor_list(lst, dtype=torch.float32):
            out = [ (x if isinstance(x, torch.Tensor) else torch.as_tensor(x)) for x in lst ]
            out = [ t.to(self.device, dtype=dtype) for t in out ]
            return out

        states_t      = torch.stack(_to_tensor_list(states,      torch.float32), dim=0)  # [B,C,D,H,W]
        next_states_t = torch.stack(_to_tensor_list(next_states, torch.float32), dim=0)  # [B,C,D,H,W]
        actions_t     = torch.as_tensor(actions, dtype=torch.long,    device=self.device).view(-1)     # [B]
        rewards_t     = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1)     # [B]
        dones_t       = torch.as_tensor(dones,   dtype=torch.bool,    device=self.device).view(-1)     # [B]

    # 前向
        q_curr_all = self.q_network(states_t)             # [B,A]
        q_next_all = self.target_network(next_states_t)   # [B,A]

        q_curr     = q_curr_all.gather(1, actions_t.unsqueeze(-1)).squeeze(-1)   # [B]
        q_next_max = q_next_all.max(dim=1)[0]                                     # [B]
        target_q   = rewards_t + (~dones_t).float() * self.gamma * q_next_max     # [B]

    # 损失（考虑 PER 权重）
        loss_elem = F.smooth_l1_loss(q_curr, target_q.detach(), reduction="none")  # [B]
        if weights is not None:
            weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device).view(-1)
            loss = (weights_t * loss_elem).mean()
        else:
            loss = loss_elem.mean()

    # 反传
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

    # 软更新
        self.align_target_model()

    # 更新 PER 优先级
        if indices is not None:
            errors = (q_curr.detach() - target_q.detach()).abs().clamp_min(1e-6).detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, errors)

    # epsilon 衰减
        if hasattr(self, "epsilon") and hasattr(self, "final_epsilon") and hasattr(self, "epsilon_decay"):
            if self.epsilon > self.final_epsilon:
                self.epsilon -= self.epsilon_decay

        return float(loss.item())
