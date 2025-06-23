import gym
import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib.animation as animation


Q = np.zeros([16, 4])


class FLWrapper(gym.Wrapper):
    def __init__(self):
        env = gym.make('FrozenLake-v1',
                       render_mode='rgb_array',
                       map_name='4x4',
                       is_slippery=False)
        super(FLWrapper, self).__init__(env)
        self.env = env
        self.step_n = 0

    def reset(self, seed=None, options=None):
        state, info = self.env.reset()
        self.step_n = 0
        return state, info

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)

        self.step_n += 1
        if self.step_n >= 200:
            done = True

        return state, reward, done, truncated, info

    def get_action(self, state, eps):
        # print("state=", state)
        if random.random() < eps:
            action = self.env.action_space.sample()
        else:
            action = Q[state].argmax()
        return action


def play_game(fl_wrapper):
    state, _ = fl_wrapper.reset()
    # print("初始state=", state)
    reword_sum = 0
    done = False

    frame_list = [fl_wrapper.render()]
    while not done:
        action = fl_wrapper.get_action(state, eps=0)
        next_state, reward, done, _, _ = fl_wrapper.step(action)
        # print("当前state=", next_state)
        state = next_state
        frame_list.append(fl_env.render())
        reword_sum += reward

    return reword_sum, frame_list


def show_game(fl_wrapper):
    reword_sum, frames = play_game(fl_wrapper)

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    im = ax.imshow(frames[0])

    def update(frame):
        print("当前frame=", frame)
        if frame >= len(frames):
            frame = len(frames) - 1
        im.set_data(frames[frame])
        return im,

    ani = animation.FuncAnimation(fig, update, frames=40, interval=200, blit=True, repeat=False)
    plt.show()


def train_model(fl_wrapper):
    eps = np.linspace(1, 0.01, 1000000)

    # 训练轮数
    for i in range(1000000):
        state, _ = fl_wrapper.reset()
        done = False

        # 一局游戏
        while not done:
            action = fl_wrapper.get_action(state, eps[i])
            next_state, reward, done, _, _ = fl_wrapper.step(action)

            # 更新Q值
            td_target = reward + 0.9 * Q[next_state].max()
            td_error = Q[state, action] - td_target
            Q[state, action] = Q[state, action] - 0.9 * td_error

            state = next_state

        if i % 500 == 0:
            mean_rwd = 0
            for _ in range(5):
                rwd, _ = play_game(fl_wrapper)
                mean_rwd += rwd
            print("第%d次训练结束后奖励: " % i, mean_rwd/5)

    np.savetxt('save/QLearning_Frozenlake', Q)


def show_env(fl_wrapper):
    fl_wrapper.reset()
    plt.figure(figsize=(3, 3))
    plt.imshow(fl_wrapper.render())
    plt.show()


if __name__ == '__main__':
    fl_env = FLWrapper()

    # 显示环境
    # show_env(fl_env)

    # 玩一局游戏并动画显示
    # show_game(fl_env)

    # result, _ = play_game(fl_env)
    # print(result)

    # 训练模型并保存
    # train_model(fl_env)

    # 加载模型并显示
    Q = np.loadtxt('save/QLearning_Frozenlake')
    show_game(fl_env)

    fl_env.close()




#
#
# def evaluate_env(fl_wrapper):
#     # agent所处的状态空间
#     print('env.observation_space=', fl_wrapper.observation_space)
#     # agent的动作空间
#     print('env.action_space=', fl_wrapper.action_space)
#
#     state = fl_wrapper.reset()
#     action = fl_wrapper.action_space.sample()
#     next_state, reward, done, truncated, info = fl_env.step(action)
#
#     print('state=', state)
#     print('action=', action)
#     print('next_state=', next_state)
#     print('reward=', reward)
#     print('done=', done)
