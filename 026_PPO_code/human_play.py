import gymnasium as gym
import pygame
import time
import numpy as np

def control_lunar_lander_with_keys(num_episodes=1):
    env = gym.make("LunarLander-v3", render_mode="human")

    pygame.init()
    pygame.display.set_caption("LunarLander Control")
    clock = pygame.time.Clock()

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        if isinstance(obs, tuple):  # 兼容新版本 Gymnasium 的 reset 返回值
            obs, _ = obs

        done = False
        ep_reward = 0

        print(f"\n🚀 Episode {episode} started. Use ← ↑ → to control.")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Quitting...")
                    env.close()
                    pygame.quit()
                    return

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action = 3  # 左推进器
            elif keys[pygame.K_RIGHT]:
                action = 1  # 右推进器
            elif keys[pygame.K_UP]:
                action = 2  # 主推进器
            else:
                action = 0  # 什么都不做

            obs, reward, terminated, truncated, info = env.step(action)
            x, y, vx, vy, angle, angle_vel, left_leg, right_leg = obs
            print(f"横向:{x}, 纵向:{y}, 水平速度:{vx}, 垂直速度:{vy}, 朝向角度:{angle}, \n角速度:{angle_vel}, 左支架:{left_leg}, 右支架:{right_leg},action:{action},得分:{reward}")
            done = terminated or truncated
            ep_reward += reward

            env.render()
            clock.tick(30)

        print(f"✅ Episode {episode} finished. Total Reward: {ep_reward:.2f}")
        time.sleep(1)  # 每回合间隔 1 秒

    env.close()
    pygame.quit()


def run_bipedal_walker_manual():
    env = gym.make("BipedalWalker-v3", render_mode="human")
    obs, _ = env.reset()

    pygame.init()
    pygame.display.set_caption("BipedalWalker Controller")

    clock = pygame.time.Clock()

    def get_action_from_keys():
        keys = pygame.key.get_pressed()
        action = np.zeros(4)

        # 控制四个关节（推荐设置成你能容易记住的组合）
        # WASD 控制左腿（0，1），方向键控制右腿（2，3）
        if keys[pygame.K_a]:  # 左髋关节前伸
            action[0] = +1
        if keys[pygame.K_z]:  # 左髋关节后拉
            action[0] = -1
        if keys[pygame.K_s]:  # 左膝关节收缩
            action[1] = +1
        if keys[pygame.K_x]:  # 左膝关节伸展
            action[1] = -1

        if keys[pygame.K_LEFT]:   # 右髋关节前伸
            action[2] = +1
        if keys[pygame.K_RIGHT]:  # 右髋关节后拉
            action[2] = -1
        if keys[pygame.K_UP]:     # 右膝关节收缩
            action[3] = +1
        if keys[pygame.K_DOWN]:   # 右膝关节伸展
            action[3] = -1

        return action

    running = True
    while running:
        # 捕获退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = get_action_from_keys()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done:
            obs, _ = env.reset()

        clock.tick(60)

    pygame.quit()
    env.close()


if __name__ == "__main__":
    control_lunar_lander_with_keys()