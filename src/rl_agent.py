import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ReplayBuffer, BatchSampler, RandomSampler
from typing import Tuple, Dict, List, Optional
from rl_env import RLOptionHedgingEnv
import random

class DQNAgent:
    """深度Q网络（DQN）代理：用于期权对冲策略学习"""
    
    def __init__(self,
                 env: RLOptionHedgingEnv,
                 gamma: float = 0.99,
                 lr: float = 1e-4,
                 batch_size: int = 64,
                 replay_buffer_size: int = 100000,
                 target_update_freq: int = 100,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 device: str = None):
        """
        初始化DQN代理
        :param env: 强化学习环境
        :param gamma: 折扣因子
        :param lr: 学习率
        :param batch_size: 批处理大小
        :param replay_buffer_size: 经验回放池大小
        :param target_update_freq: 目标网络更新频率
        :param epsilon_start: 初始探索率
        :param epsilon_end: 最终探索率
        :param epsilon_decay: 探索率衰减系数
        :param device: 计算设备（cpu/cuda）
        """
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 设备配置
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 网络初始化
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.policy_net = self._build_network().to(self.device)
        self.target_net = self._build_network().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # 初始同步
        self.target_net.eval()
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # 经验回放池
        self.replay_buffer = ReplayBuffer(
            capacity=replay_buffer_size,
            device=self.device
        )
        
        # 训练状态
        self.train_steps = 0
    
    def _build_network(self) -> nn.Module:
        """构建Q网络（3层全连接）"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作（ε-贪婪策略）
        :param state: 环境观测
        :param training: 是否训练模式（训练时探索，测试时 exploitation）
        :return: 动作索引
        """
        if training and random.random() < self.epsilon:
            # 探索：随机选择动作
            return self.env.action_space.sample()
        else:
            # 利用：选择Q值最大的动作
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, 
                        state: np.ndarray, 
                        action: int, 
                        reward: float, 
                        next_state: np.ndarray, 
                        terminated: bool, 
                        truncated: bool):
        """存储经验到回放池"""
        self.replay_buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=terminated or truncated
        )
    
    def update(self) -> float:
        """
        从回放池采样并更新Q网络
        :return: 训练损失
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0  # 回放池数据不足，不更新
        
        # 1. 采样批次数据
        batch = self.replay_buffer.sample(self.batch_size)
        states = batch['state']
        actions = batch['action'].unsqueeze(1)
        rewards = batch['reward'].unsqueeze(1)
        next_states = batch['next_state']
        dones = batch['done'].unsqueeze(1).float()
        
        # 2. 计算当前Q值（策略网络）
        current_q = self.policy_net(states).gather(1, actions)
        
        # 3. 计算目标Q值（目标网络，冻结）
        with torch.no_grad():
            next_q_max = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q_max * (1 - dones)
        
        # 4. 计算损失并反向传播
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)  # 梯度裁剪
        self.optimizer.step()
        
        # 5. 更新目标网络
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 6. 衰减探索率
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def train(self, n_episodes: int, log_freq: int = 10) -> Dict[str, List[float]]:
        """
        训练代理
        :param n_episodes: 训练轮数
        :param log_freq: 日志输出频率
        :return: 训练日志（每轮奖励、损失）
        """
        logs = {
            'episode_rewards': [],
            'episode_losses': [],
            'epsilon_history': []
        }
        
        for episode in range(n_episodes):
            episode_reward = 0.0
            episode_loss = 0.0
            step_count = 0
            
            # 重置环境
            state, _ = self.env.reset()
            
            while True:
                # 选择动作
                action = self.select_action(state, training=True)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                # 存储经验
                self.store_transition(state, action, reward, next_state, terminated, truncated)
                
                # 更新网络
                loss = self.update()
                episode_loss += loss
                
                # 累积奖励
                episode_reward += reward
                step_count += 1
                state = next_state
                
                # 检查终止条件
                if terminated or truncated:
                    break
            
            # 记录日志
            avg_loss = episode_loss / step_count if step_count > 0 else 0.0
            logs['episode_rewards'].append(episode_reward)
            logs['episode_losses'].append(avg_loss)
            logs['epsilon_history'].append(self.epsilon)
            
            # 输出日志
            if (episode + 1) % log_freq == 0:
                print(f"Episode [{episode+1}/{n_episodes}] | "
                      f"Total Reward: {episode_reward:.2f} | "
                      f"Avg Loss: {avg_loss:.4f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Steps: {step_count}")
        
        return logs
    
    def evaluate(self, n_episodes: int) -> Tuple[float, float]:
        """
        评估代理性能（无探索）
        :param n_episodes: 评估轮数
        :return: 平均奖励、奖励标准差
        """
        rewards = []
        
        for _ in range(n_episodes):
            episode_reward = 0.0
            state, _ = self.env.reset()
            
            while True:
                action = self.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                
                if terminated or truncated:
                    break
            
            rewards.append(episode_reward)
        
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        print(f"评估结果 - 平均奖励：{avg_reward:.2f} ± {std_reward:.2f}")
        return avg_reward, std_reward
    
    def save_checkpoint(self, path: str):
        """保存训练 checkpoint（网络参数、优化器、训练状态）"""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
            'replay_buffer': self.replay_buffer
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """加载训练 checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_steps = checkpoint['train_steps']
        self.replay_buffer = checkpoint['replay_buffer']
        print(f"Checkpoint loaded from {path}")

# 经验回放池实现
class ReplayBuffer:
    """简单的经验回放池"""
    def __init__(self, capacity: int, device: str):
        self.capacity = capacity
        self.device = device
        self.buffer = []
    
    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """添加经验"""
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # 先进先出
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """采样批次数据并转换为Tensor"""
        samples = random.sample(self.buffer, batch_size)
        return {
            'state': torch.tensor([s['state'] for s in samples], dtype=torch.float32).to(self.device),
            'action': torch.tensor([s['action'] for s in samples], dtype=torch.long).to(self.device),
            'reward': torch.tensor([s['reward'] for s in samples], dtype=torch.float32).to(self.device),
            'next_state': torch.tensor([s['next_state'] for s in samples], dtype=torch.float32).to(self.device),
            'done': torch.tensor([s['done'] for s in samples], dtype=torch.float32).to(self.device)
        }
    
    def __len__(self) -> int:
        return len(self.buffer)

if __name__ == "__main__":
    # 测试DQN代理
    env = RLOptionHedgingEnv()
    agent = DQNAgent(
        env=env,
        gamma=0.99,
        lr=1e-4,
        batch_size=64,
        replay_buffer_size=100000,
        target_update_freq=100,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    
    # 训练代理（10轮测试）
    print("开始训练代理...")
    logs = agent.train(n_episodes=10, log_freq=1)
    
    # 评估代理
    print("\n开始评估代理...")
    avg_reward, std_reward = agent.evaluate(n_episodes=5)
    
    # 保存checkpoint
    agent.save_checkpoint("dqn_hedging_checkpoint.pth")
    
    # 加载checkpoint（测试）
    agent.load_checkpoint("dqn_hedging_checkpoint.pth")
