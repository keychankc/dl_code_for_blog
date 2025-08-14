import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ====== Dueling DQN 网络结构 ======
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        # 公共层
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        # 状态价值分支 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)   # 输出一个标量 V(s)
        )

        # 动作优势分支 A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)  # 输出每个动作的 A(s,a)
        )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Q(s,a) = V(s) + A(s,a) - 平均优势
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


def train_dueling_dqn():
    # ====== 参数设置 ======
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
    max_episodes = 2000
    memory_size = 50000

    # ====== 初始化网络 ======
    online_net = DuelingDQN(state_dim, action_dim)
    target_net = DuelingDQN(state_dim, action_dim)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    memory = deque(maxlen=memory_size)

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

            # 存储经验
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # 训练
            if len(memory) > 1000:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                # 当前 Q 值
                q_values = online_net(states).gather(1, actions)

                # 下一个状态最大 Q 值（来自目标网络）
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                    target_q = rewards + gamma * next_q_values * (1 - dones)

                # 损失 & 更新
                loss = nn.MSELoss()(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 同步目标网络
            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(online_net.state_dict())

            if done:
                break

        # ε 衰减
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # 日志
        print(f"Episode {episode+1}/{max_episodes} | Steps: {t+1} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

    # 保存模型
    torch.save(online_net.state_dict(), "dueling_dqn_mountaincar.pth")
    print("模型已保存到 dueling_dqn_mountaincar.pth")


if __name__ == "__main__":
    train_dueling_dqn()