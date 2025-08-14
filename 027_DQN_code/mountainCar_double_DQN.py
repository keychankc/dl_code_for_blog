import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from mountaincar_DQN import DQN

def train():
    # ----- 训练参数 -----
    env = gym.make("MountainCar-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995  # 更快衰减
    lr = 1e-3
    batch_size = 64
    target_update_freq = 500
    max_episodes = 600
    memory_size = 50000

    # ----- 初始化组件 -----
    online_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
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

            # 奖励塑形：额外奖励鼓励小车向右移动
            position, velocity = next_state
            reward += abs(position - (-0.5))

            # 存储经验
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # 延迟训练：经验池 > 1000 才开始
            if len(memory) > 1000:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                # 当前Q值
                q_values = online_net(states).gather(1, actions)

                # ---- 相比 DQN Double DQN 改动部分 ----
                # 1. 主网络选择动作
                next_actions = online_net(next_states).argmax(1).unsqueeze(1)
                # 2. 目标网络评估该动作
                with torch.no_grad():
                    next_q_values = target_net(next_states).gather(1, next_actions)
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

        # 打印日志
        print(
            f"Episode {episode + 1}/{max_episodes} | Steps: {t + 1} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

    # 保存模型
    # torch.save(online_net.state_dict(), "dqn_mountaincar_fast.pth")
    torch.save(online_net.state_dict(), "double_dqn_mountaincar.pth")

    print("模型已保存到 dqn_mountaincar_fast.pth")

if __name__ == '__main__':
    train()
