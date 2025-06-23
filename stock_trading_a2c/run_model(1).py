from stock_trading_a2c.continues_trading_env import ContinuesTradingEnv
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import A2C

model = None


def run_trading(_env):
    state, _ = _env.reset()
    reword_sum = 0
    done = False

    capitals = _env.render()
    while not done:
        print("当前step=", _env.current_step, "reword=", reword_sum)
        action, _ = model.predict(state)
        next_state, reward, done, _, _ = _env.step(action)
        state = next_state
        capitals = _env.render()
        reword_sum += reward

        if _env.current_step >= _env.df.shape[0] - 1:
            break

    return reword_sum, capitals


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


data_path = 'data/510300.SH_20_22.csv'


def load_model():
    global model
    # 创建模型实例
    model = A2C.load("save/a2c_model")


if __name__ == '__main__':
    df = pd.read_csv(data_path)
    env = ContinuesTradingEnv(df=df)

    # 训练模型
    # model = SimpleTradingDNN()
    # next_model = SimpleTradingDNN()
    # train(env)

    # 加载模型并展示
    load_model()
    reword_sum, capitals = run_trading(env)
    show_results(df, capitals)
