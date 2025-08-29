import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT  # ✅ 这里导入
import pygame

"""
done: False, # 回合还没有结束
reward: 0.0, # 本次动作得到的即时奖励
action: 0, # 选择的动作编号
info: {'coins': 0, # 目前收集的金币数量
    'flag_get': False, # 是否成功到达旗子，过关标志
    'life': 2, # 当前剩余生命数
    'score': 0, # 当前分数（环境内部累计的）
    'stage': 1, # 当前关卡序号
    'status': # 马里奥的状态，例如 'small'、'big'、'fire'
    'time': 398, # 关卡剩余时间
    'world': 1, # 当前世界序号
    'x_pos': 40, # 横向位置（像素或格子）
    'y_pos': 79} # 纵向位置    
    

    
COMPLEX_MOVEMENT = [
    ['NOOP'],           # 0: 什么也不做 (No Operation)
    ['right'],          # 1: 向右走
    ['right', 'A'],     # 2: 向右 + 跳
    ['right', 'B'],     # 3: 向右 + 加速跑
    ['right', 'A', 'B'],# 4: 向右 + 跳 + 跑 (助跑跳)
    ['A'],              # 5: 原地跳
    ['left'],           # 6: 向左走
    ['left', 'A'],      # 7: 向左 + 跳
    ['left', 'B'],      # 8: 向左 + 跑
    ['left', 'A', 'B'], # 9: 向左 + 跳 + 跑
    ['down'],           # 10: 下 (蹲下，进水管/趴下)
    ['up'],             # 11: 上 (例如爬藤蔓)
]  

"""


def play():
    # 初始关卡
    world, stage = 1, 1
    env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    # 初始化 pygame
    pygame.init()
    screen = pygame.display.set_mode((256, 240))
    pygame.display.set_caption("Keyboard Play Super Mario Bros (Complex)")

    clock = pygame.time.Clock()
    action = 0  # 默认动作

    done = True
    state = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 键盘事件映射到 COMPLEX_MOVEMENT
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
            action = 7   # 左 + 跳
        elif keys[pygame.K_LEFT]:
            action = 6   # 左
        elif keys[pygame.K_RIGHT] and keys[pygame.K_SPACE] and keys[pygame.K_LSHIFT]:
            action = 4   # 右 + 跑 + 跳
        elif keys[pygame.K_RIGHT] and keys[pygame.K_LSHIFT]:
            action = 3   # 右 + 跑
        elif keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
            action = 2   # 右 + 跳
        elif keys[pygame.K_RIGHT]:
            action = 1   # 右
        elif keys[pygame.K_SPACE]:
            action = 5   # 单跳
        elif keys[pygame.K_DOWN]:
            action = 10  # 下蹲
        else:
            action = 0   # 不动

        state, reward, done, info = env.step(action)
        # print(f"action: {action}, reward: {reward}, done: {done}, info: {info}")
        env.render()

        if done:
            if info.get('flag_get', False):  # 通关自动切下一关
                world, stage = info['world'], info['stage'] + 1
                # 检查是否有下一关
                try:
                    env.close()
                    env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0")
                    env = JoypadSpace(env, COMPLEX_MOVEMENT)
                    state = env.reset()
                    print(f"切换到下一关: World {world}-{stage}")
                except:
                    print("已通关全部关卡！")
                    running = False
            else:
                state = env.reset()

        clock.tick(60)

    env.close()
    pygame.quit()


if __name__ == '__main__':
    play()