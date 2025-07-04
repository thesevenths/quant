import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 临时解决 OpenMP 冲突

import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from torch import nn
import talib  # 需安装 TA-Lib，用于技术指标计算

class TradingEnv(gym.Env):
    def __init__(self, df, close_original, start_day, end_day, window_size=10):
        super(TradingEnv, self).__init__()
        self.df = df  # 存储标准化后的OHLCV和技术指标数据
        self.close_original = close_original  # 存储原始收盘价数据
        self.start_day = start_day  # 环境开始的日期索引
        self.end_day = end_day  # 环境结束的日期索引
        self.window_size = window_size  # 滑动窗口大小，设置为10天
        self.current_day = start_day  # 当前处理的日期索引
        self.action_space = spaces.Discrete(3)  # 动作空间：0-持有，1-买入，2-卖出
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, 7), dtype=np.float32)  # 观察空间：10x7（新增MA10和RSI）

    def reset(self):
        self.current_day = self.start_day  # 重置当前日期到开始日期
        return self._get_observation()  # 返回初始观察

    def step(self, action):
        close_t = self.close_original[self.current_day]  # 当前天的收盘价
        close_t1 = self.close_original[self.current_day + 1]  # 下一天的收盘价
        if np.isnan(close_t) or np.isnan(close_t1):
            print(f"NaN detected at day {self.current_day}: close_t={close_t}, close_t1={close_t1}")
            close_t1 = close_t  # 临时用当前值替代
        position = [0, 1, -1][action]  # 根据动作确定仓位：0-持有，1-多头，-1-空头
        reward = position * (close_t1 - close_t) / close_t - 0.001  # 基本奖励减去交易成本
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0  # 临时设置为0
            print(f"Invalid reward at day {self.current_day}, set to 0")
        if reward < -0.03:  # 下跌超3%时额外惩罚
            reward -= 0.03
        if position == 1 and (close_t1 - close_t) / close_t < -0.05:  # 下跌超5%时强制卖出
            action = 2  # 卖出
            reward -= 0.05  # 减少惩罚以保留部分收益
        self.current_day += 1  # 移动到下一天
        done = self.current_day >= self.end_day  # 检查是否到达结束日期
        obs = self._get_observation() if not done else np.zeros((self.window_size, 7))  # 获取新观察或返回零向量
        print(f"Day {self.current_day}, Action {action}, Reward {reward}, Position {position}")  # 调试输出
        return obs, reward, done, {}  # 返回观察、奖励、是否结束和额外信息

    def _get_observation(self):
        start = self.current_day - self.window_size  # 窗口开始索引
        end = self.current_day  # 窗口结束索引
        window = self.df.iloc[start:end][['open', 'high', 'low', 'close', 'volume', 'MA10', 'RSI']].values  # 获取10天的OHLCV+MA10+RSI数据
        return window

def main():
    # 加载数据
    df = pd.read_csv('../data/btc_daily.csv')  # 读取比特币日数据CSV文件
    df_features = df[['open', 'high', 'low', 'close', 'volume']].copy()  # 提取OHLCV特征

    # 计算技术指标
    df_features['MA10'] = talib.SMA(df['close'].values, timeperiod=10)  # 10日移动平均线
    df_features['RSI'] = talib.RSI(df['close'].values, timeperiod=14)  # 14日相对强弱指数

    close_original = df['close'].values  # 提取原始收盘价

    # 去除 NaN 值
    df_features = df_features.dropna()
    close_original = close_original[:len(df_features)]

    # 拆分训练和测试数据，80%训练，20%测试
    total_days = len(df_features)
    train_days = int(0.8 * total_days)  # 训练数据天数
    train_df = df_features.iloc[:train_days]  # 训练数据

    # 标准化数据（包括技术指标）
    mean = train_df.mean()
    std = train_df.std()
    df_standardized = (df_features - mean) / std  # 标准化整个数据集

    # 定义训练和测试环境
    env_train = TradingEnv(df_standardized, close_original, start_day=10, end_day=train_days)  # 训练环境
    env_test = TradingEnv(df_standardized, close_original, start_day=train_days + 10, end_day=total_days - 1)  # 测试环境，结束前一天避免越界

    # 检查模型是否存在，若存在则加载，否则训练并保存
    model_path = 'ppo_trading.zip'  # 模型保存路径
    if os.path.exists(model_path):
        model = PPO.load(model_path)  # 加载已有模型
    else:
        model = PPO('MlpPolicy', env_train, policy_kwargs={'net_arch': [256, 128, 64], 'activation_fn': nn.SiLU, 'ortho_init': True}, verbose=1)  # 创建PPO模型，自定义3层结构
        model.learn(total_timesteps=400000)  # 训练40万步
        model.save(model_path)  # 保存训练完成的模型
        print("Model structure:", model.policy)  # 打印模型结构
        print("Total parameters:", sum(p.numel() for p in model.policy.parameters()))  # 打印总参数量

    # 测试阶段
    obs = env_test.reset()  # 重置测试环境
    done = False
    portfolio_value = 1  # 初始投资价值
    portfolio_values = [1]  # 记录投资价值的列表，初始值为1
    while not done:
        action, _states = model.predict(obs, deterministic=True)  # 预测动作
        obs, reward, done, info = env_test.step(action)  # 执行动作并获取反馈
        portfolio_value *= (1 + reward)  # 更新投资价值
        portfolio_values.append(portfolio_value)  # 记录当前投资价值

    # 绘制累计收益曲线
    dates = df['datetime'].iloc[env_test.start_day + 1: env_test.end_day]
    btc_price = close_original[env_test.start_day + 1: env_test.end_day] / close_original[env_test.start_day]  # 归一化BTC价格
    strategy_plot = portfolio_values[1:-1]  # 策略投资价值，去掉初始值和最后一个多余值以匹配dates长度

    plt.plot(dates, btc_price, label='BTC Price (Normalized)')  # 绘制归一化BTC价格曲线
    plt.plot(dates, strategy_plot, label='Strategy Portfolio Value')  # 绘制策略投资价值曲线
    plt.xlabel('Date')  # X轴标签
    plt.ylabel('Value')  # Y轴标签
    plt.title('Cumulative Reward Curve (Portfolio Value) on Test Data')  # 标题
    plt.legend()  # 显示图例
    plt.xticks(rotation=45)  # 旋转X轴标签45度
    plt.tight_layout()  # 自动调整布局
    plt.show()  # 显示图表

if __name__ == '__main__':
    main()  # 运行主函数