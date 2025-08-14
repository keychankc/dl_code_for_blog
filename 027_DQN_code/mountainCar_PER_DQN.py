import gymnasium as gym
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
from mountaincar_DQN import DQN

# ========== 优先经验回放类 ==========
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        """存储一条经验，并赋予最大优先级"""
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """按优先级采样"""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # 重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = torch.FloatTensor(batch[0])
        actions = torch.LongTensor(batch[1]).unsqueeze(1)
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1)
        next_states = torch.FloatTensor(batch[3])
        dones = torch.FloatTensor(batch[4]).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        """更新采样到的样本的优先级"""
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

# ========== 训练函数 ==========
def train_per_dqn():
    env = gym.make("MountainCar-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    lr = 1e-3
    batch_size = 64
    target_update_freq = 500
    max_episodes = 600
    memory_size = 50000
    alpha = 0.6  # 优先程度
    beta_start = 0.4
    beta_frames = 100000  # beta 从 0.4 逐渐增加到 1.0

    online_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=lr)

    memory = PrioritizedReplayBuffer(memory_size, alpha)
    total_steps = 0

    for episode in range(max_episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(200):
            total_steps += 1

            # ε-贪婪策略
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = online_net(state_tensor)
                    action = q_values.argmax().item()

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 奖励塑形
            position, velocity = next_state
            reward += abs(position - (-0.5))

            # 存储经验（初始优先级 = 最大值）
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # PER 采样并训练
            if len(memory) > 1000:
                beta = min(1.0, beta_start + (1.0 - beta_start) * total_steps / beta_frames)
                states, actions, rewards, next_states, dones, indices, weights = memory.sample(batch_size, beta)

                # 当前 Q
                q_values = online_net(states).gather(1, actions)

                # 目标 Q
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                    target_q = rewards + gamma * next_q_values * (1 - dones)

                # TD 误差
                td_errors = target_q - q_values
                loss = (weights * td_errors.pow(2)).mean()

                # 更新网络
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 更新优先级
                new_priorities = td_errors.abs().detach().numpy() + 1e-6
                memory.update_priorities(indices, new_priorities)

            # 同步目标网络
            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(online_net.state_dict())

            if done:
                break

        # ε 衰减
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {episode+1}/{max_episodes} | Steps: {t+1} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

    torch.save(online_net.state_dict(), "per_dqn_mountaincar.pth")
    print("模型已保存到 per_dqn_mountaincar.pth")


if __name__ == "__main__":
    train_per_dqn()