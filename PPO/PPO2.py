import os
import logging
from pathlib import Path

import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 临时解决 OpenMP 冲突

import gymnasium as gym
from gymnasium import spaces  # ✅ 必须从 gymnasium 导入！
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from torch import nn
import talib

# 配置日志
log_dir = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在目录
log_file = os.path.join(log_dir, "trading_log.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # 保存到文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

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

    def reset(self, seed=None, options=None):
        # 设置随机种子（可选，你的交易环境可能不需要随机性）
        if seed is not None:
            np.random.seed(seed)
        # 重置当前日期
        self.current_day = self.start_day
        # 返回 observation 和 info 字典
        return self._get_observation(), {}

    def step(self, action):
        close_t = self.close_original[self.current_day]  # 当前天的收盘价
        close_t1 = self.close_original[self.current_day + 1]  # 下一天的收盘价
        if np.isnan(close_t) or np.isnan(close_t1):
            logging.error(f"NaN detected at day {self.current_day}: close_t={close_t}, close_t1={close_t1}")
            close_t1 = close_t  # 临时用当前值替代
        position = [0, 1, -1][action]  # 根据动作确定仓位：0-持有，1-多头，-1-空头
        reward = position * (close_t1 - close_t) / close_t - 0.001  # 基本奖励减去交易成本
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0  # 临时设置为0
            logging.error(f"Invalid reward at day {self.current_day}, set to 0")
        if reward < -0.03:  # 下跌超3%时额外惩罚
            reward -= 0.03
        if position == 1 and (close_t1 - close_t) / close_t < -0.05:  # 下跌超5%时强制卖出
            action = 2  # 卖出
            reward -= 0.05  # 减少惩罚以保留部分收益
        
        self.current_day += 1  # 移动到下一天
        # done = self.current_day >= self.end_day  # 检查是否到达结束日期
        # obs = self._get_observation() if not done else np.zeros((self.window_size, 7))  # 获取新观察或返回零向量
        terminated = self.current_day >= self.end_day
        truncated = False
        obs = self._get_observation() if not terminated else np.zeros((self.window_size, 7))
        logging.info(f"Day {self.current_day}, Action {action}, Reward {reward}, Position {position}")  # 调试输出
        return obs, reward, terminated, truncated, {}  # 返回观察、奖励、是否结束和额外信息

    def _get_observation(self):
        start = self.current_day - self.window_size  # 窗口开始索引
        end = self.current_day  # 窗口结束索引
        window = self.df.iloc[start:end][['open', 'high', 'low', 'close', 'volume', 'MA10', 'RSI']].values  # 获取10天的OHLCV+MA10+RSI数据
        return window

def calculate_max_drawdown(portfolio_values):
    """
    计算最大回撤
    :param portfolio_values: 投资组合价值的时间序列列表
    :return: 最大回撤值（百分比）
    """
    max_drawdown = 0.0
    peak = portfolio_values[0]  # 初始峰值
    
    for value in portfolio_values:
        if value > peak:
            peak = value  # 更新历史峰值
        else:
            # 计算当前回撤比例（相对于历史峰值）
            drawdown = (peak - value) / peak  
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    
    return max_drawdown

def main():

    import stable_baselines3
    print("SB3 Version:", stable_baselines3.__version__)

    # 加载数据
    csv_path = Path(__file__).parent / 'btc_daily.csv'
    df = pd.read_csv(csv_path)
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
    model_path = Path(__file__).parent /'ppo_trading2.zip'  # 模型保存路径
    if os.path.exists(model_path):
        model = PPO.load(model_path)  # 加载已有模型
    else:
        model = PPO(
            'MlpPolicy', 
            env_train, 
            policy_kwargs={
                'net_arch': [256, 128, 64], # 网络过大容易记住noise导致过拟合
                'activation_fn': nn.SiLU, 
                'ortho_init': True,
                'optimizer_class': torch.optim.AdamW, 
                'optimizer_kwargs': {'weight_decay': 1e-4, 'eps': 1e-5}
            },
            learning_rate=2.5e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1
        )
        model.learn(total_timesteps=400000)  # 训练40万步
        model.save(model_path)  # 保存训练完成的模型
        logging.info("Model structure: %s", model.policy)  # 打印模型结构
        logging.info("Total parameters: %d", sum(p.numel() for p in model.policy.parameters()))  # 打印总参数量

    # 测试阶段
    # obs = env_test.reset()  # 重置测试环境
    obs, info = env_test.reset()
    done = False
    portfolio_value = 1  # 初始投资价值
    portfolio_values = [1]  # 记录投资价值的列表，初始值为1
    while not done:
        action, _states = model.predict(obs, deterministic=True)  # 预测动作
        obs, reward, terminated, truncated, info = env_test.step(action)  # 执行动作并获取反馈
        done = terminated or truncated

        portfolio_value *= (1 + reward)  # 更新投资价值
        portfolio_values.append(portfolio_value)  # 记录当前投资价值

    # 计算最大回撤
    max_drawdown = calculate_max_drawdown(portfolio_values)
    logging.info(f"最大回撤 (Max Drawdown): {max_drawdown * 100:.2f}%")

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

    # 找到最大回撤的起始和结束索引
    peak_idx = np.argmax(np.maximum.accumulate(portfolio_values) == np.array(portfolio_values))
    trough_idx = np.argmin(portfolio_values[peak_idx:]) + peak_idx

    # 在图表中添加标注
    plt.plot(dates, btc_price, label='BTC Price (Normalized)')
    plt.plot(dates, strategy_plot, label='Strategy Portfolio Value')
    plt.axvspan(dates[peak_idx], dates[trough_idx], color='red', alpha=0.3, label=f'Max Drawdown {max_drawdown * 100:.2f}%')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()  # 运行主函数