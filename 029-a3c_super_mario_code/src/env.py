import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import subprocess as sp
from nes_py.wrappers import JoypadSpace
from gym import Wrapper
import cv2
import numpy as np
from gym.spaces import Box


def create_train_env(world, stage, action_type, output_path=None):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))
    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None
    if action_type == "right":
        actions = RIGHT_ONLY
    elif action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    env = JoypadSpace(env, actions) # 把复杂的按钮组合，封装成离散动作编号，方便后续强化学习训练或键盘控制
    env = CustomReward(env, monitor) # 自定义奖励信号
    env = CustomSkipFrame(env) # 跳帧+状态堆叠
    return env, env.observation_space.shape[0], len(actions)


class Monitor:
    """
        把环境的画面录制下来并保存成视频文件
    """
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "80", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        try:
            self.pipe.stdin.write(image_array.tobytes())
        except BrokenPipeError:
            print("Warning: ffmpeg pipe closed")

class CustomReward(Wrapper):
    """
    把环境原始观测处理成灰度 84×84 的输入，同时修改奖励信号，让训练更稳定
    """
    def __init__(self, env=None, monitor=None):
        super(CustomReward, self).__init__(env)
        # 把环境原始观测（通常是 (240, 256, 3) 的彩色帧）修改成 (1, 84, 84) 的灰度图
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        if self.monitor:
            self.monitor.record(state) # 录屏

        state = process_frame(state) # 游戏画面预处理

        reward += (info["score"] - self.curr_score) / 40. # 奖励塑形
        self.curr_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        return state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        return process_frame(self.env.reset()) # 返回初始状态

def process_frame(frame):
    """
    把原始游戏画面做预处理，变成适合神经网络输入的数据
    """
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))


class CustomSkipFrame(Wrapper):
    """
    跳帧 + 状态堆叠，让智能体不用每一帧都观察和决策
    1.	跳帧：一个动作连续执行多帧（skip 帧），减少环境计算量，提高训练效率。
	2.	堆叠帧：将多帧图像堆叠起来作为状态输入，使智能体能感知运动方向和速度。
    """
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84)) # 堆叠4帧灰度图像
        self.skip = skip

    def step(self, action):
        total_reward = 0
        states = []
        state, reward, done, info = self.env.step(action)
        for i in range(self.skip):
            if not done:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                states.append(state)
            else:
                states.append(state)
        states = np.concatenate(states, 0)[None, :, :, :]
        return states.astype(np.float32), reward, done, info

    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
        return states.astype(np.float32)