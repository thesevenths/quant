import torch
import random
import gym
from matplotlib import pyplot as plt
import matplotlib.animation as animation

# 数据样本池
data_set = []
model = None
next_model = None


def get_action(state):
    # 通过神经网络, 得到一个动作
    state = torch.FloatTensor(state).reshape(1, 4)
    return model(state).argmax().item()


# 更新样本池，向样本池中添加N条数据, 删除M条最古老的数据
def update_data(_env):
    old_count = len(data_set)

    # 玩到新增了N个数据为止
    while len(data_set) - old_count < 200:
        # 初始化游戏
        state = _env.reset()
        state = state[0]

        # 玩到游戏结束为止
        done = False
        while not done:
            # 根据当前状态得到一个动作
            action = get_action(state)

            # 执行动作,得到反馈
            next_state, reward, done, _, _ = _env.step(action)

            # 记录数据样本
            data_set.append((state, action, reward, next_state, done))

            # 更新游戏状态,开始下一个动作
            state = next_state

    # 数据上限,超出时从最古老的开始删除
    while len(data_set) > 1_0000:
        data_set.pop(0)


# 获取一批数据样本
def get_sample():
    # 从样本池中采样64条数据
    samples = random.sample(data_set, 64)

    # [b, 4] state
    state = torch.FloatTensor([i[0] for i in samples]).reshape(-1, 4)
    # [b, 1] action
    action = torch.LongTensor([i[1] for i in samples]).reshape(-1, 1)
    # [b, 1] reword
    reward = torch.FloatTensor([i[2] for i in samples]).reshape(-1, 1)
    # [b, 4] next_state
    next_state = torch.FloatTensor([i[3] for i in samples]).reshape(-1, 4)
    # [b, 1] done
    done = torch.LongTensor([i[4] for i in samples]).reshape(-1, 1)

    return state, action, reward, next_state, done


def get_value(state, action):
    # 使用状态计算出动作的logits
    # [b, 4] -> [b, 2]
    value = model(state)

    # 根据实际使用的action取出每一个值
    # 这个值就是模型评估的在该状态下,执行动作的分数
    # 在执行动作前,显然并不知道会得到的反馈和next_state
    # 所以这里不能也不需要考虑next_state和reward
    # [b, 2] -> [b, 1]
    value = value.gather(dim=1, index=action)

    return value


def get_target(reward, next_state, done):
    # 上面已经把模型认为的状态下执行动作的分数给评估出来了
    # 下面使用next_state和reward计算真实的分数
    # 针对一个状态,它到底应该多少分,可以使用以往模型积累的经验评估
    # 这也是没办法的办法,因为显然没有精确解,这里使用延迟更新的next_model评估

    # 使用next_state计算下一个状态的分数
    # [b, 4] -> [b, 2]
    with torch.no_grad():
        target = next_model(next_state)

    # 取所有动作中分数最大的
    # [b, 2] -> [b, 1]
    target = target.max(dim=1)[0]
    target = target.reshape(-1, 1)

    # 下一个状态的分数乘以一个系数,相当于权重
    target *= 0.98

    # 如果next_state已经游戏结束,则next_state的分数是0
    # 因为如果下一步已经游戏结束,显然不需要再继续玩下去,也就不需要考虑next_state了.
    # [b, 1] * [b, 1] -> [b, 1]
    target *= (1 - done)

    # 加上reward就是最终的分数
    # [b, 1] + [b, 1] -> [b, 1]
    target += reward

    return target


def evaluate(_env):
    state = _env.reset()[0]
    reward_sum = 0
    done = False
    while not done:
        action = get_action(state)
        state, reward, done, _, _ = _env.step(action)
        reward_sum += reward
    return reward_sum


def train(_env):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = torch.nn.MSELoss()

    # 训练N次
    for epoch in range(500):
        # 更新N条数据
        update_data(_env)

        # 每次更新过数据后,学习N次
        for i in range(200):
            # 采样一批数据
            state, action, reward, next_state, over = get_sample()

            # 计算一批样本的value和target
            value = get_value(state, action)
            target = get_target(reward, next_state, over)

            # 更新参数
            loss = loss_fn(value, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 把model的参数复制给next_model
            if (i + 1) % 10 == 0:
                next_model.load_state_dict(model.state_dict())

        if epoch % 50 == 0:
            print(epoch, len(data_set), sum([evaluate(_env) for _ in range(5)]) / 5)

    torch.save(model, 'save/DQN_CartPole')


def load_model():
    global model
    # 创建模型实例
    model = torch.load('save/DQN_CartPole', weights_only=False)
    model.eval()


def play_game(_env):
    state = _env.reset()[0]
    reword_sum = 0
    done = False

    frame_list = [_env.render()]
    while not done:
        action = get_action(state)
        next_state, reward, done, _, _ = _env.step(action)
        # print("当前state=", next_state)
        state = next_state
        frame_list.append(_env.render())
        reword_sum += reward

        print(len(frame_list))
        if len(frame_list) > 200:
            break

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


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='rgb_array', max_episode_steps=200)

    # 训练模型
    # model = CartPoleDNN()
    # next_model = CartPoleDNN()
    # train(env)

    # 加载模型并展示
    load_model()
    show_game(env)


