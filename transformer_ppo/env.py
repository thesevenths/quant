import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.spaces import Box, Discrete


class BTCTradingEnv(gym.Env):
    def __init__(self, data_path='btc_daily.csv', seq_len=10, initial_balance=10000000): # initial_balance必须比close price大， 否则无法买入
        super().__init__()
        self.df = pd.read_csv(data_path)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.seq_len = seq_len
        self.initial_balance = initial_balance
        self.current_step = 0
        self.position = 0  # 0: no position空仓, 1: long多头
        self.balance = initial_balance
        self.holding = 0  # BTC held
        self.max_steps = len(self.df) - seq_len - 1  # 确保不越界
        self.buy_price = 0  # 记录买入价格

        self.observation_space = Box(  # seq_len*5
            low=-np.inf, high=np.inf, shape=(seq_len, 5), dtype=np.float32
        )
        self.action_space = Discrete(3)  # 0: hold, 1: buy, 2: sell

        # Normalize features
        self.features = self.df[['open', 'high', 'low', 'close', 'volume']].values
        self.features = (self.features - self.features.mean(axis=0)) / (self.features.std(axis=0) + 1e-8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.seq_len
        self.balance = self.initial_balance
        self.holding = 0
        self.position = 0
        self.buy_price = 0
        return self._get_observation(), {}

    def _get_observation(self):  # 最近seq_len步的历史数据
        return self.features[self.current_step - self.seq_len:self.current_step]

    def step(self, action):
        current_price = self.df['close'].iloc[self.current_step]
        next_price = self.df['close'].iloc[self.current_step + 1]

        reward = 0
        if action == 1 and self.position == 0:  # 空舱买入
            if self.balance >= current_price:  # 余额大于close价格
                self.holding = self.balance / current_price  # 全部买入
                self.buy_price = current_price  # 记录买入价格
                reward += (next_price - current_price) / current_price * 100
                self.balance = 0
                self.position = 1
                print(f"[Step {self.current_step}] Action: {action}, position: {self.position} , buy Reward: {reward:.5f}, Balance: {self.balance:.2f}, Holding: {self.holding}, current Price: {current_price:.2f}, buy Price: {self.buy_price:.2f}, next Price: {next_price:.2f}")

        elif action == 0 and self.position == 0: # 空仓hold：空仓踏空了? 空仓躲过下跌了？
            reward +=  (current_price - next_price) / current_price *100
            print(f"[Step {self.current_step}] Action: {action}, position: {self.position} ,  空仓hold等待 reward: {reward:.5f}, Balance: {self.balance:.2f}, Holding: {self.holding}, current Price: {current_price:.2f}, next Price: {next_price:.2f}")
        
        elif action == 2 and self.position == 1:  # 全仓卖出
            self.balance = self.holding * current_price  # 卖出btc增加balance
            # reward = (self.balance - self.initial_balance) / self.initial_balance * 100 # 应该是卖出价格比买入价格高才奖励，否则这次交易是亏损的
            # 这一步卖出；如果明天跌价，说明今天卖对了要奖励，否则处罚；但这样做会让模型短视：只考虑明天的价格，不看长远、频繁交易
            # reward += (current_price - next_price) / current_price * 20
            reward += (current_price - self.buy_price) / self.buy_price * 100  # 卖出价格比买入高有奖励，否则惩罚
            self.holding = 0
            self.position = 0
            print(f"[Step {self.current_step}] Action: {action}, position: {self.position} , sell Reward: {reward:.5f}, Balance: {self.balance:.2f}, Holding: {self.holding}, current Price: {current_price:.2f}, buy Price: {self.buy_price:.2f}, next Price: {next_price:.2f}")
            self.buy_price = 0

        elif action == 0 and self.position == 1: # 全仓等待
             # reward += (next_price - current_price) / current_price * 100  # btc涨价了要奖励；跌价了就处罚; 这种方案可能导致短视，只看眼前不看长远
            reward += (current_price - self.buy_price) / self.buy_price * 100 # current_price只要比buy_price高就奖励，否则惩罚
            print(f"[Step {self.current_step}] Action: {action}, position: {self.position} , hold reward: {reward:.5f}, Balance: {self.balance:.2f}, Holding: {self.holding}, current Price: {current_price:.2f}, buy Price: {self.buy_price:.2f}, next Price: {next_price:.2f}")
        else:  # action =2 and position = 0: 空仓卖出    action = 1 and position = 1： 全仓买入 都是不可能的组合，轻微处罚
            reward -= 0.005  # Reduced penalty
            print(f"[Step {self.current_step}] Action: {action}, position: {self.position} , wrong combine Reward: {reward:.5f}, Balance: {self.balance:.2f}, Holding: {self.holding}, current Price: {current_price:.2f}, buy Price: {self.buy_price:.2f}, next Price: {next_price:.2f}")

        # if self.position == 1:  # 持有BTC不动
        #     # reward += (next_price - current_price) / current_price * 100  # btc涨价了要奖励；跌价了就处罚; 这种方案可能导致短视，只看眼前不看长远
        #     reward += (current_price - self.buy_price) / self.buy_price * 100 # current_price只要比buy_price高就奖励，否则惩罚
        #     print(f"  [Step {self.current_step}] Action: {action}, position: {self.position} , hold reward: {reward:.5f}, Balance: {self.balance:.2f}, Holding: {self.holding}, current Price: {current_price:.2f}, next Price: {next_price:.2f}")


        # print(f"[Step {self.current_step}] Action: {action}, position: {self.position} , total Reward: {reward:.5f}, Balance: {self.balance:.2f}, Holding: {self.holding}, current Price: {current_price:.2f}, buy Price: {self.buy_price:.2f}, next Price: {next_price:.2f}")

        self.current_step += 1
        done = self.current_step >= self.max_steps or self.balance < 0 # 不能等于0，否则全仓买入episode就结束了
        truncated = False

        return self._get_observation(), reward, done, truncated, {}  # 下一步观测值

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Holding: {self.holding:.4f} BTC")


def make_env(data_path='btc_daily.csv', seq_len=10):
    return BTCTradingEnv(data_path, seq_len)