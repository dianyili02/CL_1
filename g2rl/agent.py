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
            action_map=None):           # 动作索引映射（网络->环境）

        self.device = device
        self.action_space = action_space
        self.replay_buffer = PrioritizedReplayBuffer(replay_buffer_size, alpha)
        
        self.tau = tau
        self.q_network = model
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

    def store(self, state: Dict[str, Any], action: int, reward: float, next_state: Dict[str, Any], terminated: bool):
        state_cache = state['view_cache']
        next_state_cache = next_state['view_cache']
        transition = (state_cache, action, reward, next_state_cache, terminated)
        state_tensor = torch.tensor(np.array(state_cache)).float().to(self.device)
        next_state_tensor = torch.tensor(np.array(next_state_cache)).float().to(self.device)
        with torch.no_grad():
            curr_Q = self.q_network(state_tensor.unsqueeze(0)).squeeze(0)[action]
            next_Q = self.target_network(next_state_tensor.unsqueeze(0)).max(1)[0].item()
        target = reward + self.gamma * next_Q * (1 - terminated)
        error = abs(curr_Q.item() - target)
        self.replay_buffer.add(error, transition)

    def align_target_model(self):
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    # def act(self, state: Dict[str, Any]) -> int:
    #     state = state['view_cache']
    #     state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
    #     if random.random() <= self.epsilon:
    #         return random.choice(self.action_space)
    #     with torch.no_grad():
    #         q_values = self.q_network(state)
    #     return torch.argmax(q_values).item()
    # 测试用
    def act(self, state):
        # 1) 从 obs 里拿时序视野并重排通道
        V = state['view_cache']  # 形状 [T,H,W,C]
        # 安全转换 + 重排到训练时顺序（我们探针推断为 [障碍, 自车, 目标, 其他] = (1,0,3,2)）
        V = np.asarray(V, dtype=np.float32)[..., list(self.use_channels)]  # [T,H,W,C']
        # 转成 [1,C,T,H,W]
        x = torch.from_numpy(V).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)

        # 2) 前向 & 选网络动作
        with torch.no_grad():
            q = self.q_network(x)
            net_action = int(torch.argmax(q, dim=1).item())

        # 3) 网络动作 -> 环境动作（如你之前校准出的 [0,2,1,3,4]）
        env_action = self.action_map[net_action]
        return env_action

    

    def retrain(self, batch_size: int) -> float:
        if len(self.replay_buffer) < batch_size:
            return
        
        samples, indices, weights = self.replay_buffer.sample(batch_size, self.beta)
        states, actions, rewards, next_states, dones = zip(*samples)
        
        states = torch.tensor(np.array(states)).float().to(self.device)
        next_states = torch.tensor(np.array(next_states)).float().to(self.device)
        actions = torch.tensor(actions).long().to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device)
        weights = torch.tensor(weights).float().to(self.device)

        curr_Q = self.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_Q = self.target_network(next_states).max(1)[0]
        expected_Q = rewards + self.gamma * next_Q * (1 - dones)
        loss = (weights * F.mse_loss(curr_Q, expected_Q.detach(), reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.align_target_model()
        
        errors = torch.abs(curr_Q - expected_Q).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, errors)
        
        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_decay
            
        return loss.item()
