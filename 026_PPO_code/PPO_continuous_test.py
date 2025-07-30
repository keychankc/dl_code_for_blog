import gymnasium as gym
from PPO_continuous import PPO, Memory
from PIL import Image
import torch
import pygame
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_bipedal_walker_gif():
    ############## Hyperparameters ##############
    env_name = "BipedalWalker-v3"
    env = gym.make(env_name, render_mode="rgb_array")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    n_episodes = 1  # 为节省帧数，只录制一个 episode
    max_timesteps = 1500

    filename = f"PPO_continuous_{env_name}.pth"
    directory = "./"

    action_std = 0.5
    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99
    lr = 0.0003
    betas = (0.9, 0.999)
    #############################################

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    ppo.policy_old.load_state_dict(torch.load(directory + filename, weights_only=True))

    for ep in range(1, n_episodes + 1):
        ep_reward = 0
        state, _ = env.reset()

        frames = []

        for t in range(max_timesteps):
            action = ppo.select_action(state, memory)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            frame = env.render()
            img = Image.fromarray(frame)
            frames.append(img)

            if terminated or truncated:
                break

        print(f'Episode: {ep}\tReward: {int(ep_reward)}')

        # 保存为 GIF
        os.makedirs("./gif", exist_ok=True)
        gif_path = f'./gif/bipedal_walker_ep{ep}.gif'
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
            duration=40,  # 每帧持续时间，毫秒
            loop=0
        )
        print(f'GIF saved to {gif_path}')
    env.close()

def render_bipedal_walker():
    ############## Hyperparameters ##############
    env_name = "BipedalWalker-v3"
    env = gym.make(env_name, render_mode="human")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    n_episodes = 3
    max_timesteps = 1500

    filename = f"PPO_continuous_{env_name}.pth"
    directory = "./"

    action_std = 0.5
    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99
    lr = 0.0003
    betas = (0.9, 0.999)
    #############################################

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    ppo.policy_old.load_state_dict(torch.load(directory + filename, weights_only=True))

    for ep in range(1, n_episodes + 1):
        ep_reward = 0
        state, _ = env.reset()

        for t in range(max_timesteps):
            action = ppo.select_action(state, memory)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            try:
                env.render()
                pygame.event.pump()  # 防止 Pygame 卡死或被系统关闭
            except pygame.error:
                print("⚠️ Pygame window was closed. Skipping rendering.")
                break

            if terminated or truncated:
                break

        print(f'Episode: {ep}\tReward: {int(ep_reward)}')

    env.close()  # 在所有 episode 后统一关闭


if __name__ == '__main__':
    render_bipedal_walker()