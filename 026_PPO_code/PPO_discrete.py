import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    # 经验缓存类
    def __init__(self):
        self.actions = []  # 存储动作
        self.states = []  # 存储状态
        self.logprobs = []  # 存储旧策略下动作的对数概率
        self.rewards = []  # 存储奖励
        self.is_terminals = []  # 存储是否为终止状态标志

    def clear_memory(self):
        # 清空缓存（每次 update 后调用）
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    # 创建一个 Actor-Critic 架构：即同时学习动作策略（Actor）和状态价值（Critic）
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # Actor网络（策略网络）：用于输出每个动作的概率分布
        # 输入：环境状态 输出：每个动作的概率
        # 结构：输入层 → 两个隐藏层（Tanh激活）→ 输出层（Softmax概率）
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(), # Tanh 激活函数，给网络增加非线性表达能力，帮助网络学习更复杂的特征
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)  # 输出动作概率分布
        )

        # Critic网络（价值网络）：用于输出当前状态的价值（state value）
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1) # 输出状态价值
        )

    # forward() 不需要定义，因为我们单独定义了 act() 和 evaluate() 方法。
    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        # 用当前策略选择动作
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)  # 得到动作概率
        dist = Categorical(action_probs)  # 按照给定的概率分布来进行采样
        action = dist.sample() # 按概率采样动作

        # 将动作与 log 概率存入 memory，用于后续更新
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        # 用于评估旧策略下动作的概率和价值
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy() # 概率分布熵（鼓励探索）

        state_value = self.value_layer(state) # 状态价值

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    # 更新策略
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, k_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = k_epochs

        # 初始化策略网络（新旧）、优化器、损失函数
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    # 策略更新核心函数
    def update(self, memory):
        # 使用 GAE 估计每个状态的回报
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 对回报进行标准化（加快收敛）
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # 取出旧的数据（detach 防止反向传播）
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # K 次 epoch 的 PPO 更新:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # 概率比率 r_t
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # PPO 剪切损失
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # 构造 PPO 总损失（策略 loss + value loss + 熵）
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # 执行梯度更新
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 更新旧策略参数
        self.policy_old.load_state_dict(self.policy.state_dict())


def main():
    ############## 参数配置 ##############
    env_name = "LunarLander-v3"
    env = gym.make(env_name) # 创建游戏环境
    state_dim = env.observation_space.shape[0] # 状态空间的维度，输入层的神经元数量，等于环境中每个状态的特征数
    action_dim = 4 # 动作空间的维度，输出层的神经元数量，等于可选动作的数量（离散动作空间时）
    n_latent_var = 64  # 隐藏层的神经元数量，神经网络隐藏层的宽度，影响模型容量，常用 64、128
    lr = 0.002 # 学习率 控制每次参数更新的步长，影响训练速度和稳定性
    betas = (0.9, 0.999) # Adam 优化器的动量参数，通常为 (0.9, 0.999)
    gamma = 0.99  # 奖励折扣因子，控制未来奖励的衰减程度，越接近 1 越重视长期奖励
    k_epochs = 4  # 每次更新策略时的训练轮数，每收集一批数据后，策略网络要训练多少次，如 4（离散动作），80（连续动作）
    eps_clip = 0.2  # PPO 裁剪参数，限制新旧策略概率比的变化范围，防止策略更新过大，保证训练稳定
    random_seed = None # 设置随机种子

    render = False
    solved_reward = 230  # 完成训练任务的奖励阈值
    log_interval = 20  # 日志打印间隔
    max_episodes = 50000  # 最大训练回合数
    max_timesteps = 300  # 每个 episode 的最大步数
    update_timestep = 2000  # 策略更新步数间隔
    #############################################

    # 设置随机种子
    if random_seed is not None:
        torch.manual_seed(random_seed)
        env.reset(seed=random_seed)
        print(f"PyTorch随机数:{torch.rand(1).item()}, 环境初始状态:{env.reset()[0][:2]}")

    # 初始化 PPO、Memory、变量
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, k_epochs, eps_clip)

    # logging variables
    running_reward = 0
    total_length = 0
    timestep = 0

    finished_step = 0

    # training loop
    for i_episode in range(1, max_episodes + 1): # 一共玩多少局
        state, _ = env.reset()  # 初始化
        for t in range(max_timesteps): # 每一局最多能走多少步
            finished_step = t
            timestep += 1

            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            # next_state：执行动作后的新状态 reward：本步获得的奖励 terminated：是否因为任务完成/失败而结束 truncated：是否因为达到最大步数等外部原因而结束
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state

            # 收集 reward 和 done
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # 满足 update_timestep 时，更新策略并清空经验
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            running_reward += reward
            if render:
                env.render()
            if done:
                break

        total_length += finished_step # 累计每个 episode（回合）实际运行的步数

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval * solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            break

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(total_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            total_length = 0

if __name__ == '__main__':
    main()