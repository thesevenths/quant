# encoding = utf-8
from simple_trading_env import SimpleTradingEnv
import pandas as pd


data_path = 'data/510300.SH_20_24.csv'


def create_env():
    df = pd.read_csv(data_path)
    env = SimpleTradingEnv(df=df, transaction_fee=0)
    return env


if __name__ == '__main__':
    steps = [0, 1, 1, 0, 0, 1, 0, 1, 0]
    trading_env = create_env()
    state = trading_env.reset()
    for action in steps:
        # action = trading_env.action_space.sample()
        nex_state, reward, done, _, info = trading_env.step(action)
        print(f'action:{action}, reward:{reward}, done:{done}, info:{info}')
        # trading_env.render()
        if done:
            break
