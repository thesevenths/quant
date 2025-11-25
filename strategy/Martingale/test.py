import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fetch_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, 'btc_usdt_2y_1h.csv')
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"文件 {csv_file} 不存在！")
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['timestamp'] = df['timestamp'].astype('int64')
    return df.sort_values('datetime').reset_index(drop=True)

# 简单信号生成：假设我们用“前一小时涨跌”作为信号
# to do:使用更稳健的信号（如均线、RSI、机器学习等）
def generate_signals(df, lookback=1):
    """
    简单动量信号：如果前1小时上涨，则下注“涨”
    """
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['signal'] = df['return'].shift(1) > 0  # 上涨则下一轮做多
    return df

def martingale_backtest(df, initial_capital=1_000_000, base_bet=10_000):
    """
    马丁格尔策略：
    - 初始每笔下注 base_bet
    - 如果亏损，下一笔下注 = 2 * 上一笔
    - 如果盈利，重置为 base_bet
    """
    position = 0  # 当前持仓（+1 表示多头，但我们用资金流模拟）
    bet_size = base_bet
    capital = initial_capital
    capital_history = [capital]
    bet_history = []
    
    for i, row in df.iterrows():
        if pd.isna(row['signal']):
            capital_history.append(capital)
            bet_history.append(0)
            continue
        
        # 模拟：在当前小时开始时下注，按下一小时价格结算
        if i + 1 >= len(df):
            break
        future_return = df.loc[i + 1, 'return']
        
        if row['signal']:  # 看涨
            pnl = bet_size * future_return
        else:  # 看跌（简化：反向）
            pnl = -bet_size * future_return
        
        capital += pnl
        capital_history.append(capital)
        bet_history.append(bet_size)
        
        # 更新下一次下注大小
        if pnl < 0:  # 亏损 → 加倍
            bet_size = min(bet_size * 2, capital * 0.9)  # 防止下注超过资金
        else:  # 盈利 → 重置
            bet_size = base_bet
        
        # 安全机制：资本≤0 则爆仓
        if capital <= 0:
            capital_history.extend([0] * (len(df) - len(capital_history) + 1))
            break
    
    return np.array(capital_history[:-1]), np.array(bet_history)

def anti_martingale_backtest(df, initial_capital=1_000_000, base_bet=10_000):
    """
    反马丁格尔策略：
    - 盈利后加倍下注，亏损后重置为 base_bet
    """
    bet_size = base_bet
    capital = initial_capital
    capital_history = [capital]
    bet_history = []
    
    for i, row in df.iterrows():
        if pd.isna(row['signal']):
            capital_history.append(capital)
            bet_history.append(0)
            continue
        
        if i + 1 >= len(df):
            break
        future_return = df.loc[i + 1, 'return']
        
        if row['signal']:
            pnl = bet_size * future_return
        else:
            pnl = -bet_size * future_return
        
        capital += pnl
        capital_history.append(capital)
        bet_history.append(bet_size)
        
        if pnl > 0:  # 盈利 → 加倍
            bet_size = min(bet_size * 2, capital * 0.9)
        else:  # 亏损 → 重置
            bet_size = base_bet
        
        if capital <= 0:
            capital_history.extend([0] * (len(df) - len(capital_history) + 1))
            break
    
    return np.array(capital_history[:-1]), np.array(bet_history)

def buy_and_hold(df, initial_capital=1_000_000):
    """买入持有策略作为基准"""
    returns = df['close'].pct_change().fillna(0)
    cum_returns = (1 + returns).cumprod()
    return initial_capital * cum_returns

def main():
    df = fetch_data()
    df = generate_signals(df)
    
    # 回测
    mart_cap, mart_bets = martingale_backtest(df)
    anti_cap, anti_bets = anti_martingale_backtest(df)
    bh_cap = buy_and_hold(df)
    
    # 对齐长度
    min_len = min(len(mart_cap), len(anti_cap), len(bh_cap))
    mart_cap = mart_cap[:min_len]
    anti_cap = anti_cap[:min_len]
    bh_cap = bh_cap[:min_len]
    df_plot = df.iloc[:min_len].copy()
    
    # 绘图
    plt.figure(figsize=(14, 7))
    plt.plot(df_plot['datetime'], mart_cap, label='Martingale', alpha=0.8)
    plt.plot(df_plot['datetime'], anti_cap, label='Anti-Martingale', alpha=0.8)
    plt.plot(df_plot['datetime'], bh_cap, label='Buy & Hold', alpha=0.8)
    plt.title('BTC Martingale vs Anti-Martingale vs Buy & Hold (2 Years)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (CNY/USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('strategy_comparison.png')
    plt.show()
    
    # 输出最终收益
    initial = 1_000_000
    print(f"\n初始资金: {initial:,.0f}")
    print(f"马丁格尔最终: {mart_cap[-1]:,.0f} (收益: {(mart_cap[-1]/initial - 1)*100:.2f}%)")
    print(f"反马丁格尔最终: {anti_cap[-1]:,.0f} (收益: {(anti_cap[-1]/initial - 1)*100:.2f}%)")
    print(f"买入持有最终: {bh_cap.iloc[-1]:,.0f} (收益: {(bh_cap.iloc[-1]/initial - 1)*100:.2f}%)")
    
    # 风险提示
    print("\n" + "="*60)
    print("⚠️  风险警告：")
    print("1. 马丁格尔策略在连续亏损时下注指数增长，极易爆仓。")
    print("2. 本回测假设：无滑点、无手续费、可无限分割 BTC。")
    print("3. 实盘中市场可能长期单边（如持续下跌），马丁格尔会快速亏光本金。")
    print("4. 反马丁格尔虽更安全，但仍依赖信号质量，可能高买低卖。")
    print("✅ 建议：仅用于学习，不要实盘！使用止损、风控、多样化策略。")
    print("="*60)

if __name__ == "__main__":
    main()