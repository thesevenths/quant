from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from project_ai.learn_sb3.pendulum.pendulum_env import PendulumEnv
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from pendulum.model import MODEL


model = MODEL


def play_game(env):
    state, _ = env.reset()
    print("初始state=", state)
    reword_sum = 0
    done = False

    frame_list = [env.render()]
    while not done:
        env: PendulumEnv = env
        action, _ = model.predict(state)
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


if __name__ == '__main__':

    # 测试

    model = A2C.load("save/a2c_model")

    # 创建环境
    mc_env = PendulumEnv()

    # 玩一局游戏并动画显示
    show_game(mc_env)

