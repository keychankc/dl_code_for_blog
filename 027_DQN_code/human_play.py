import gym
import pygame
import numpy as np
import time

def run_mountain_car_manual():
    # 初始化 pygame
    pygame.init()
    screen = pygame.display.set_mode((400, 100))
    pygame.display.set_caption("Manual Control - MountainCarContinuous-v0")

    # 初始化环境
    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    obs, _ = env.reset()
    clock = pygame.time.Clock()

    running = True
    action = np.array([0.0], dtype=np.float32)  # 初始动作

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 按键检测
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = np.array([-1.0], dtype=np.float32)
        elif keys[pygame.K_RIGHT]:
            action = np.array([1.0], dtype=np.float32)
        else:
            action = np.array([0.0], dtype=np.float32)

        # 执行动作
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"当前状态: {obs}, 当前动作: {action}, 得分：{reward}")
        env.render()

        # 每秒最多 30 帧
        clock.tick(30)

        if terminated or truncated:
            print("Episode finished. Resetting environment.")
            obs, _ = env.reset()
            time.sleep(1)

    env.close()
    pygame.quit()

if __name__ == "__main__":
    run_mountain_car_manual()