import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.spaces import Box, Discrete

class BTCTradingEnv(gym.Env):
    def __init__(self, data_path='btc_daily.csv', seq_len=10, initial_balance=10000):
        super().__init__()
        self.df = pd.read_csv(data_path)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.seq_len = seq_len
        self.initial_balance = initial_balance
        self.current_step = 0
        self.position = 0  # 0: no position, 1: long
        self.balance = initial_balance
        self.holding = 0  # BTC held
        self.max_steps = len(self.df) - seq_len - 1

        self.observation_space = Box(
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
        return self._get_observation(), {}

    def _get_observation(self):
        return self.features[self.current_step - self.seq_len:self.current_step]

    def step(self, action):
        current_price = self.df['close'].iloc[self.current_step]
        next_price = self.df['close'].iloc[self.current_step + 1]

        reward = 0
        if action == 1 and self.position == 0:  # Buy
            if self.balance >= current_price:
                self.holding = self.balance / current_price
                self.balance = 0
                self.position = 1
        elif action == 2 and self.position == 1:  # Sell
            self.balance = self.holding * current_price
            reward = (self.balance - self.initial_balance) / self.initial_balance * 100  # Amplify reward
            self.holding = 0
            self.position = 0
        else:  # Hold penalty
            reward -= 0.01  # Small penalty to encourage trading
        if self.position == 1:
            reward += (next_price - current_price) / current_price * 10  # Amplify price change

        self.current_step += 1
        done = self.current_step >= self.max_steps or self.balance <= 0
        truncated = False

        return self._get_observation(), reward, done, truncated, {}

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Holding: {self.holding:.4f} BTC")

def make_env(data_path='btc_daily.csv', seq_len=10):
    return BTCTradingEnv(data_path, seq_len)