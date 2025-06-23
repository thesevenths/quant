import gym
import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib.animation as animation


class PendulumEnv(gym.Wrapper):

    def __init__(self):
        env = gym.make('Pendulum-v1', render_mode='rgb_array')
        super().__init__(env)
        self.step_n = 0
        self.env = env

    def reset(self, seed=None, options=None):
        state, info = self.env.reset()
        self.step_n = 0
        return state, info

    def step(self, action):
        # print(f"Action before step: {action}, type: {type(action)}")
        if isinstance(action, (list, np.ndarray)):
            action = np.array(action).flatten()[0]
        else:
            action = float(action)
        state, reward, done, truncated, info = self.env.step(action)

        self.step_n += 1
        if self.step_n >= 200:
            done = True

        return state, reward, done, truncated, info


def play_game(env):
    state, _ = env.reset()
    print("初始state=", state)
    reword_sum = 0
    done = False

    frame_list = [env.render()]
    while not done:
        env: PendulumEnv = env
        action = env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)
        print("当前state=", next_state)
        state = next_state
        frame_list.append(env.render())
        reword_sum += reward

    return reword_sum, frame_list


def show_game(env):
    reword_sum, frames = play_game(env)

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    im = ax.imshow(frames[0])

    def update(frame):
        # print("当前frame=", frame)
        if frame >= len(frames):
            frame = len(frames) - 1
        im.set_data(frames[frame])
        return im,

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=10, blit=True, repeat=False)
    plt.show()


def show_env(env):
    state, _ = env.reset()
    print("初始state=", state)
    plt.figure(figsize=(3, 3))
    plt.imshow(env.render())
    plt.show()


if __name__ == '__main__':
    mc_env = PendulumEnv()

    # 显示环境
    # show_env(mc_env)

    # 玩一局游戏并动画显示
    show_game(mc_env)

    # result, _ = play_game(fl_env)
    # print(result)

    # 训练模型并保存
    # train_model(fl_env)

    # 加载模型并显示
    # Q = np.loadtxt('save/QLearning_Frozenlake')
    # show_game(fl_env)
