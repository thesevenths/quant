import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any


class ContinuesTradingEnv(gym.Env):
    """
        股票交易强化学习环境
        动作空间: 标量，[0, 1] 之间的仓位比例 (0表示空仓，1表示满仓)
        状态空间: [历史价格窗口, 仓位, 账户余额]
    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 df: pd.DataFrame,
                 window_size: int = 60,
                 initial_capital: float = 10000.0,
                 transaction_fee: float = 0.001  # 默认交易额的千分之一
                 ):
        """
        初始化环境
        :param df: 包含OHLCV的DataFrame (Open, High, Low, Close, Volume)
        :param window_size: 观测历史窗口大小
        :param initial_capital: 初始资金
        :param transaction_fee: 交易手续费率
        """
        super(ContinuesTradingEnv, self).__init__()

        # 数据参数
        self.df = df.dropna().reset_index(drop=True)   # 标的数据
        self.window_size = window_size                 # 窗口
        self.current_step = None                       # 与时间下标对应
        self.current_date = None

        # 交易参数
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee

        # 状态信息
        self.capital = None         # 当前账户市值
        self.cash = None            # 当前现金
        self.market_value = None    # 当前市值
        self.shares = None          # 当前持仓份额
        self.share_position = 0.0   # 当前持仓比例（占当前账户价值比例）
        self.history = []           # 记录交易历史

        # 动作空间: 标量，[0, 1] 之间的仓位比例 (0表示空仓，1表示满仓)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # 观测空间: (窗口*价格特征 + 份额 + 当前账户价值)
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(window_size * 5 + 2,),  # OHLCV + shares + capital
            dtype=np.float32
        )

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """
        重置环境状态
        :return: 初始观测值
        """
        super().reset(seed=seed)

        self.current_step = self.window_size - 1       # 从0开始计数，真正的交易从 self.window_size - 1 这一个时间步开始
        self.initial_capital = self.initial_capital
        self.capital = self.initial_capital
        self.cash = self.initial_capital
        self.market_value = 0
        self.shares = 0
        self.share_position = 0.0
        self.current_date = self.df.loc[self.current_step, 'Date']
        self.history = []
        return self._next_observation(), {}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行动作
        :param action: 仓位比例
        :return: (observation, reward, terminated, truncated, info)
        """
        truncated = False
        # 检查是否结束
        if self.current_step >= len(self.df) - 1:
            return self._next_observation(), 0, False, True, {}

        # 获取当前价格
        current_price = self._get_current_price()

        # 执行交易动作
        self._take_action(action[0], current_price)

        # 更新状态
        self.current_step += 1
        self.current_date = self.df.loc[self.current_step, 'Date']
        self.capital = self.cash + self.shares * self._get_current_price()

        # 计算奖励 (可修改为更复杂的奖励函数)
        reward = self._calculate_reward()

        # 检查是否终止
        terminated = self.capital <= 0

        # 记录交易信息
        info = {
            'date_trans': 'from %s to %s' % (self.df.loc[self.current_step-1, 'Date'], self.df.loc[self.current_step, 'Date']),
            'step_trans': 'from %d to %d' % (self.current_step-1, self.current_step),
            'now_capital': self.capital,
            'now_cash': self.cash,
            'now_shares': self.shares,
            'last_price': self.df.loc[self.current_step-1, 'Close'],
            'now_price': self._get_current_price()
        }
        self.history.append(info)

        return self._next_observation(), reward, terminated, truncated, info

    def _take_action(self, target_position: float, current_price: float) -> None:
        """ 执行具体交易动作 """

        target_shares = (self.capital / current_price) * target_position  # 目标持仓份额
        # 计算差额
        delta_shares = target_shares - self.shares

        if delta_shares > 0:  # 需要买入
            max_buyable = self.cash // (current_price * (1 + self.transaction_fee))
            buy_shares = min(delta_shares, max_buyable)
            if buy_shares > 0:
                self.shares += buy_shares
                cost = buy_shares * current_price * (1 + self.transaction_fee)
                self.cash -= cost
                self.market_value = self.shares * current_price
                self.capital = self.cash + self.market_value
                self.share_position = self.market_value / self.capital

        elif delta_shares < 0:  # 需要卖出
            sell_shares = min(-delta_shares, self.shares)
            if sell_shares > 0:
                revenue = sell_shares * current_price * (1 - self.transaction_fee)
                self.cash += revenue
                self.shares -= sell_shares
                self.market_value = self.shares * current_price
                self.capital = self.cash + self.market_value
                self.share_position = self.market_value / self.capital

    def _calculate_reward(self) -> float:
        """ 计算奖励值 """
        # 基础奖励：净值变化
        reward = (self.capital - self.initial_capital) / self.initial_capital * 100  # 百分比变化

        # 风险惩罚（可选）
        if self.shares > 0:
            price_window = self.df['Close'][
                           self.current_step + 1 - self.window_size:self.current_step+1
                           ]
            volatility = price_window.pct_change().std()
            reward -= 0.1 * volatility * 100  # 波动率惩罚

        return reward

    def _get_current_price(self) -> float:
        """ 获取当前时间步的收盘价 """
        return self.df.loc[self.current_step, 'Close']

    def _next_observation(self) -> np.ndarray:
        """ 构建观测值 """

        if self.current_step >= 60:
            pass

        # 获取价格窗口数据 (OHLCV)：包含当前时间步，以及之前的window_size内的OHLCV数据
        obv_bars = self.df.iloc[self.current_step + 1
                                - self.window_size:self.current_step+1][['Open', 'High', 'Low', 'Close', 'Volume']]
        price_window = obv_bars.values.flatten()

        # 标准化处理价格系列 z-score
        mean_price = obv_bars['Close'].mean()
        std_price = obv_bars['Close'].std()
        norm_price = (price_window - mean_price) / std_price
        norm_shares = self.share_position
        norm_capital = self.capital / self.initial_capital

        # 组合观测值
        obs = np.concatenate((
            norm_price,
            np.array([norm_shares, norm_capital])
        )).astype(np.float32)

        return obs

    def render(self, mode: str = 'human') -> list:
        """ 可视化 """
        if mode == 'human':
            capitals = [x['now_capital'] for x in self.history]
            return capitals

