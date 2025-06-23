import sys
sys.path.append('/codes')
from stock_trading_a2c.continues_trading_env import ContinuesTradingEnv
import pandas as pd


data_path = '/stock_trading_a2c/data/510300.SH_20_24.csv'


def create_env():
    df = pd.read_csv(data_path)
    env = ContinuesTradingEnv(df=df, transaction_fee=0)
    return env


if __name__ == '__main__':
    steps = [0.2, 0.8, 0.5, 0.0, 0.1, 0.4, 0.3, 1.0, 0.0]
    trading_env = create_env()
    state, _ = trading_env.reset(seed=42)
    for action in steps:
        next_state, reward, terminated, truncated, info = trading_env.step(action)
        print(f'action:{action}, reward:{reward}, done:{terminated}, info:{info}')
        if terminated or truncated:
            break
