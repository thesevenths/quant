# encoding=utf-8

import gym
from matplotlib import pyplot as plt
import matplotlib.animation as animation


def play_game(_env):
    _env.reset()
    reword_sum = 0
    done = False

    frame_list = [_env.render()]
    while not done:
        action = _env.action_space.sample()
        next_state, reward, done, _, _ = _env.step(action)
        print("当前state=", next_state)
        state = next_state
        frame_list.append(_env.render())
        reword_sum += reward

    return reword_sum, frame_list


def show_game(_env):
    reword_sum, frames = play_game(_env)

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    im = ax.imshow(frames[0])

    def update(frame):
        if frame >= len(frames):
            # 如果帧数不够，就用最后一帧
            frame = len(frames) - 1
        im.set_data(frames[frame])
        return im,

    ani = animation.FuncAnimation(fig, update, frames=200, blit=True, repeat=False)
    plt.show()


def show_info(_env):
    print('env.observation_space=', _env.observation_space)
    print('env.action_space=', _env.action_space)

    state = _env.reset()
    action = _env.action_space.sample()
    next_state, reward, done, _, _ = _env.step(action)

    print('state=', state)
    print('action=', action)
    print('next_state=', next_state)
    print('reward=', reward)
    print('done=', done)


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='rgb_array', max_episode_steps=200)
    show_game(_env=env)
    # show_info(_env=env)

