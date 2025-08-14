import gymnasium as gym
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import imageio
import numpy as np
from mountainCar_dueling_DQN import DuelingDQN

# ----- DQN网络 -----
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # 两层隐层 MLP，将连续状态映射到每个离散动作的 Q 值
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x) # 输出形状：[batch, action_dim]，即 Q(s, ·)

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

                # ---- DQN 部分 ----
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

        # 打印日志
        print(
            f"Episode {episode + 1}/{max_episodes} | Steps: {t + 1} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

    # 保存模型
    # torch.save(online_net.state_dict(), "dqn_mountaincar_fast.pth")
    torch.save(online_net.state_dict(), "double_dqn_mountaincar.pth")

    print("模型已保存到 dqn_mountaincar_fast.pth")

def evaluate_to_gif():
    # === 1. 加载训练好的模型 ===
    model = DQN(state_dim=2, action_dim=3)
    model.load_state_dict(torch.load("dqn_mountaincar_fast.pth"))
    model.eval()

    # 注意：这里 render_mode 改成 "rgb_array" 方便截图
    env = gym.make("MountainCar-v0", render_mode="rgb_array")

    num_episodes = 10
    success_count = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        step = 0
        frames = []  # 存放每一帧画面

        while not done:
            # --- 获取当前帧画面 ---
            frame = env.render()  # render() 返回的是 RGB 数组
            frames.append(frame)

            # 模型选择动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_tensor)
            action = q_values.argmax().item()

            # 执行动作
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1

        # 成功计数
        if state[0] >= 0.5:
            success_count += 1

        print(f"Episode {episode + 1} | Steps: {step} | Final Pos: {state[0]:.2f}")

        # --- 保存 gif ---
        gif_path = f"mountaincar_ep{episode+1}.gif"
        imageio.mimsave(gif_path, frames, fps=30)  # 30 FPS 动画
        print(f"保存动画到 {gif_path}")

    success_rate = success_count / num_episodes * 100
    print(f"成功率: {success_rate:.2f}%")

    env.close()

def evaluate_model(model, model_path, episodes=100):
    """评估模型在 MountainCar-v0 上的成功率"""
    env = gym.make("MountainCar-v0")  # 无渲染模式
    model.load_state_dict(torch.load(model_path))
    model.eval()

    success_count = 0
    total_steps = []
    final_positions = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        steps = 0

        while not done:
            with torch.no_grad():
                q_values = model(torch.FloatTensor(state).unsqueeze(0))
            action = q_values.argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        if state[0] >= 0.5:
            success_count += 1
        total_steps.append(steps)
        final_positions.append(state[0])

    env.close()
    success_rate = success_count / episodes * 1000
    avg_steps = np.mean(total_steps)
    avg_pos = np.mean(final_positions)

    return success_rate, avg_steps, avg_pos

def evaluate_dqn():
    dqn_result = evaluate_model(DQN(state_dim=2, action_dim=3), "dqn_mountaincar_fast.pth")
    double_dqn_result = evaluate_model(DQN(state_dim=2, action_dim=3), "double_dqn_mountaincar.pth")
    dueling_dqn_result = evaluate_model(DuelingDQN(state_dim=2, action_dim=3), "dueling_dqn_mountaincar.pth")
    per_dqn_result = evaluate_model(DQN(state_dim=2, action_dim=3), "per_dqn_mountaincar.pth")

    print("===== 成功率对比（1000 回合） =====")
    print(
        f"DQN         | 成功率: {dqn_result[0]:.2f}% | 平均步数: {dqn_result[1]:.2f} | 平均终点位置: {dqn_result[2]:.2f}")
    print(
        f"Double DQN  | 成功率: {double_dqn_result[0]:.2f}% | 平均步数: {double_dqn_result[1]:.2f} | 平均终点位置: {double_dqn_result[2]:.2f}")
    print(
        f"Dueling DQN  | 成功率: {dueling_dqn_result[0]:.2f}% | 平均步数: {dueling_dqn_result[1]:.2f} | 平均终点位置: {dueling_dqn_result[2]:.2f}")
    print(
        f"PER DQN      | 成功率: {per_dqn_result[0]:.2f}% | 平均步数: {per_dqn_result[1]:.2f} | 平均终点位置: {per_dqn_result[2]:.2f}")
if __name__ == '__main__':
    # train()
    # evaluate()
    evaluate_dqn()

