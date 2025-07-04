import torch
import random
import pandas as pd
import gym
from simple_trading_env import SimpleTradingEnv
from dnn import SimpleTradingDNN
import matplotlib.pyplot as plt

# 数据样本池
data_set = []
model = None
next_model = None

STATE_DIM = 302
ACTION_DIM = 3
TIME_RANGE = 200


def get_action(state):
    # 通过神经网络, 得到一个动作
    state = torch.FloatTensor(state).reshape(1, STATE_DIM)
    return model(state).argmax().item()


# 更新样本池，向样本池中添加N条数据, 删除M条最古老的数据
def update_data(_env):
    # 初始化游戏
    state = _env.reset()

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
    while len(data_set) > TIME_RANGE:
        data_set.pop(0)


# 获取一批数据样本
def get_sample():
    # 从样本池中采样64条数据
    samples = random.sample(data_set, 64)

    # [b, STATE_DIM] state
    state = torch.FloatTensor([i[0] for i in samples]).reshape(-1, STATE_DIM)
    # [b, 1] action
    action = torch.LongTensor([i[1] for i in samples]).reshape(-1, 1)
    # [b, 1] reword
    reward = torch.FloatTensor([i[2] for i in samples]).reshape(-1, 1)
    # [b, STATE_DIM] next_state
    next_state = torch.FloatTensor([i[3] for i in samples]).reshape(-1, STATE_DIM)
    # [b, 1] done
    done = torch.LongTensor([int(i[4]) for i in samples]).reshape(-1, 1)

    return state, action, reward, next_state, done


def get_value(state, action):
    # 使用状态计算出动作的logits
    # [b, STATE_DIM] -> [b, 2]
    value = model(state)

    # 根据实际使用的action取出每一个值
    # 这个值就是模型评估的在该状态下,执行动作的分数
    # 在执行动作前,显然并不知道会得到的反馈和next_state
    # 所以这里不能也不需要考虑next_state和reward
    # [b, 302] -> [b, 3]
    value = value.gather(dim=1, index=action)

    return value


def get_target(reward, next_state, done):
    # 上面已经把模型认为的状态下执行动作的分数给评估出来了
    # 下面使用next_state和reward计算真实的分数
    # 针对一个状态,它到底应该多少分,可以使用以往模型积累的经验评估
    # 这也是没办法的办法,因为显然没有精确解,这里使用延迟更新的next_model评估

    # 使用next_state计算下一个状态的分数
    # [b, 302] -> [b, 3]
    with torch.no_grad():
        target = next_model(next_state)

    # 取所有动作中分数最大的
    # [b, 302] -> [b, 3]
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
    state = _env.reset()
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

    torch.save(model, 'save/DQN_simple_trading')


def load_model():
    global model
    # 创建模型实例
    model = torch.load('save/DQN_simple_trading', weights_only=False)
    model.eval()


def run_trading(_env):
    state = _env.reset()
    reword_sum = 0
    done = False

    capitals = _env.render()
    while not done:
        action = get_action(state)
        next_state, reward, done, _, _ = _env.step(action)
        # print("当前state=", next_state)
        state = next_state
        capitals = _env.render()
        reword_sum += reward

        if _env.current_step >= 608:
            pass

    return reword_sum, capitals


data_path = 'data/510300.SH_20_22.csv'


def show_results(_df, _capitals):
    # 绘制价格曲线和净值曲线
    plt.figure(figsize=(12, 6))

    # 价格曲线
    hs300 = _df['Close'] / _df['Close'][0]
    nvs = [1 for _ in range(0, 60)] + list(_capitals / _capitals[0])
    plt.subplot(1, 1, 1)
    plt.plot(hs300, label='300', color='green')
    plt.plot(nvs, label='Net Value', color='red')
    plt.xticks(rotation=45)  # 旋转日期标签以便更好地显示
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv(data_path)
    env = SimpleTradingEnv(df=df, transaction_fee=0)

    # 训练模型
    # model = SimpleTradingDNN()
    # next_model = SimpleTradingDNN()
    # train(env)

    # 加载模型并展示
    load_model()
    reword_sum, capitals = run_trading(env)
    show_results(df, capitals)

