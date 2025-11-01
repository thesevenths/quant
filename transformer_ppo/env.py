# env.py
import gymnasium as gym
import numpy as np
import pandas as pd
import pandas_ta as ta
from gymnasium.spaces import Box, Discrete

class BTCTradingEnv(gym.Env):
    """
    日线比特币交易环境（多头 + 空仓）——用于 RL 训练。
    关键设计：
      - 仅使用历史和当前已知数据（无 future price 泄露）；
      - 奖励基于净资产变化率；
      - 支持交易成本、滑点、止损／止盈、仓位限制、杠杆等扩展；
      - 可扩展更多指标与风险控制机制。
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 data_path='btc_daily.csv',
                 seq_len=10,
                 initial_balance=10_000_000,
                 trading_fee=0.001,
                 slippage_pct=0.0005,
                 max_hold_days=None,
                 use_leverage=1.0,
                 stop_loss_pct=None,
                 take_profit_pct=None):
        super().__init__()
        # -- 参数说明 --
        # data_path: 历史数据路径，必须按时间升序
        # seq_len: 观察窗口长度（以日为单位）
        # initial_balance: 初始资金
        # trading_fee: 每次买／卖的比例手续费
        # slippage_pct: 交易滑点估计（按比例计算）
        # max_hold_days: 最长持仓天数限制（超过则强制平仓）
        # use_leverage: 杠杆倍数（如 1.0 = 全仓，无杠杆）
        # stop_loss_pct: 持仓触发止损比例（如 -0.05 = 5%亏损触发）
        # take_profit_pct: 持仓触发止盈比例（如 +0.10 = 10%盈利触发）

        self.df = pd.read_csv(data_path)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.seq_len = seq_len
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.slippage_pct = slippage_pct
        self.max_hold_days = max_hold_days
        self.use_leverage = use_leverage
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # --- 技术指标计算 ---
        # 趋势指标
        self.df['sma20'] = ta.sma(self.df['close'], length=20)
        self.df['ema12'] = ta.ema(self.df['close'], length=12)
        self.df['ema26'] = ta.ema(self.df['close'], length=26)
        # 动量指标
        self.df['rsi14'] = ta.rsi(self.df['close'], length=14)
        macd = ta.macd(self.df['close'], fast=12, slow=26, signal=9)
        self.df['macd'] = macd['MACD_12_26_9']
        self.df['macd_signal'] = macd['MACDs_12_26_9']
        # 波动率／区间指标
        bb = ta.bbands(self.df['close'], length=20, std=2)
        self.df['bb_upper'] = bb['BBU_20_2.0']
        self.df['bb_lower'] = bb['BBL_20_2.0']
        self.df['atr14'] = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=14)
        # 成交量／能量指标
        self.df['obv'] = ta.obv(self.df['close'], self.df['volume'])
        self.df['ad'] = ta.ad(self.df['high'], self.df['low'], self.df['close'], self.df['volume'])

        # 特征列选定
        self.use_features = [
            'open','high','low','close','volume',
            'sma20','ema12','ema26','rsi14',
            'macd','macd_signal','bb_upper','bb_lower','atr14','obv','ad'
        ]
        self.df[self.use_features] = self.df[self.use_features].fillna(0)
        # 归一化／标准化（注意：理想情况下应只用训练集统计量，而非整个数据集）
        self.features = self.df[self.use_features].values
        self.features = (self.features - self.features.mean(axis=0)) / (self.features.std(axis=0) + 1e-8)

        # 定义 observation 和 action 空间
        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(seq_len, len(self.use_features)),
            dtype=np.float32
        )
        self.action_space = Discrete(3)  # 0=hold／空仓，1=buy(long), 2=sell(exit long)

        # 内部状态初始化
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 从 seq_len 开始，保证有足够历史数据
        self.current_step = self.seq_len
        self.balance = self.initial_balance
        self.holding = 0.0    # 持有的币数
        self.position = 0      # 0：无仓，1：有多头仓
        self.buy_price = 0.0
        self.hold_days = 0     # 持仓天数计数
        self.last_total_value = self.balance + self.holding * self.df['close'].iloc[self.current_step-1]
        obs = self._get_observation()
        return obs, {}

    def _get_observation(self):
        # 返回最近 seq_len 步（不含未来）标准化特征
        return self.features[self.current_step-self.seq_len : self.current_step].astype(np.float32)

    def step(self, action):
        done = False
        truncated = False

        current_price = self.df['close'].iloc[self.current_step]
        # 执行动作
        if action == 1 and self.position == 0:
            # 买入全部（支持杠杆）
            available = self.balance * self.use_leverage
            units = available / current_price
            # 滑点扣减
            units = units * (1.0 - self.slippage_pct)
            self.holding = units
            self.buy_price = current_price * (1.0 + self.slippage_pct)
            self.balance = self.balance - (units * current_price * (1.0 + self.trading_fee))
            self.position = 1
            self.hold_days = 0
        elif action == 2 and self.position == 1:
            # 卖出全部
            sale_value = self.holding * current_price * (1.0 - self.slippage_pct)
            self.balance = sale_value * (1.0 - self.trading_fee)
            self.holding = 0.0
            self.position = 0
            self.buy_price = 0.0
            self.hold_days = 0
        # action == 0 or illegal => hold

        # 更新持仓天数
        if self.position == 1:
            self.hold_days += 1

        # 检查止损／止盈／持仓超时强制平仓
        if self.position == 1:
            current_return = (current_price - self.buy_price) / (self.buy_price + 1e-8)
            if (self.stop_loss_pct is not None and current_return <= self.stop_loss_pct) or \
               (self.take_profit_pct is not None and current_return >= self.take_profit_pct) or \
               (self.max_hold_days is not None and self.hold_days >= self.max_hold_days):
                # 强制卖出
                sale_value = self.holding * current_price * (1.0 - self.slippage_pct)
                self.balance = sale_value * (1.0 - self.trading_fee)
                self.holding = 0.0
                self.position = 0
                self.buy_price = 0.0
                self.hold_days = 0

        # 前一步净资产
        prev_total = self.last_total_value

        # 推进时间
        self.current_step += 1
        if self.current_step >= len(self.df):
            done = True

        # 计算当前总资产
        new_price = self.df['close'].iloc[min(self.current_step, len(self.df)-1)]
        total_value = self.balance + self.holding * new_price

        # 计算 reward：基于净资产变化率，而不是基于价格，避免look-ahead bias数据泄露
        reward = (total_value - prev_total) / (prev_total + 1e-8) * 100.0
        self.last_total_value = total_value

        info = {
            'holding': self.holding,
            'balance': self.balance,
            'total_value': total_value,
            'current_price': current_price,
            'position': self.position,
            'hold_days': self.hold_days
        }

        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, truncated, info

    def render(self, mode='human'):
        price = self.df['close'].iloc[self.current_step]
        print(f"Step: {self.current_step}, Date: {self.df['datetime'].iloc[self.current_step].date()}, "
              f"Balance: {self.balance:.2f}, Holding: {self.holding:.4f}, "
              f"TotalAsset: {self.last_total_value:.2f}, Price: {price:.2f}, Position: {self.position}, "
              f"HoldDays: {self.hold_days}")

def make_env(data_path='btc_daily.csv', seq_len=10, initial_balance=10_000_000, **kwargs):
    return BTCTradingEnv(data_path=data_path, seq_len=seq_len, initial_balance=initial_balance, **kwargs)
