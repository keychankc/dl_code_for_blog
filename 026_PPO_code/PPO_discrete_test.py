import gymnasium as gym
from PPO_discrete import PPO, Memory
from PIL import Image
import torch
import pygame
import os

def save_lunar_lander_gif():
    ############## Hyperparameters ##############
    env_name = "LunarLander-v3"
    env = gym.make(env_name, render_mode="rgb_array")  # 改为 rgb_array 以获取图像帧

    state_dim = env.observation_space.shape[0]
    action_dim = 4
    n_latent_var = 64
    lr = 0.0007
    betas = (0.9, 0.999)
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2
    #############################################

    n_episodes = 1  # 只录制 1 个 episode，避免 GIF 太长
    max_timesteps = 300

    filename = f"PPO_{env_name}.pth"
    directory = "./"

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    ppo.policy_old.load_state_dict(torch.load(os.path.join(directory, filename)), strict=False)

    for ep in range(1, n_episodes + 1):
        ep_reward = 0
        state, _ = env.reset()

        frames = []

        for t in range(max_timesteps):
            action = ppo.policy_old.act(state, memory)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            # 采集帧图像
            frame = env.render()
            frames.append(Image.fromarray(frame))

            if terminated or truncated:
                break

        print(f'Episode: {ep}\tReward: {int(ep_reward)}')

        # 保存 gif 动画
        os.makedirs("./gif", exist_ok=True)
        gif_path = f'./gif/lunar_lander_ep{ep}.gif'
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=40,  # 每帧持续时间，单位毫秒
            loop=0
        )
        print(f"GIF saved to {gif_path}")

    env.close()


def start_lunar_lander():
    ############## Hyperparameters ##############
    env_name = "LunarLander-v3"
    env = gym.make(env_name, render_mode="human")

    state_dim = env.observation_space.shape[0]
    action_dim = 4
    n_latent_var = 64
    lr = 0.0007
    betas = (0.9, 0.999)
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2
    #############################################

    n_episodes = 3
    max_timesteps = 300

    filename = f"PPO_{env_name}.pth"
    directory = "./"

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    ppo.policy_old.load_state_dict(torch.load(os.path.join(directory, filename)), strict=False)

    for ep in range(1, n_episodes + 1):
        ep_reward = 0
        state, _ = env.reset()

        for t in range(max_timesteps):
            action = ppo.policy_old.act(state, memory)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            try:
                env.render()
                pygame.event.pump()  # 防止窗口卡死
            except pygame.error:
                print("Render window was closed unexpectedly.")
                break

            if terminated or truncated:
                break

        print(f'Episode: {ep}\tReward: {int(ep_reward)}')

    env.close()  # 正确地在最后统一关闭环境

if __name__ == '__main__':
    start_lunar_lander()