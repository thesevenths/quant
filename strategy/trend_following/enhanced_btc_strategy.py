import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fetch_data():
    """Load BTC/USDT 1-hour data."""
    if '__file__' in globals():
        script_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        script_dir = os.getcwd()
    csv_file = os.path.join(script_dir, 'btc_usdt_2y_1h.csv')
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File {csv_file} not found!")
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)

def generate_better_signals(df):
    """趋势跟踪信号：只做多"""
    df = df.copy()
    df['ma_20'] = df['close'].rolling(20, min_periods=1).mean()
    df['ma_50'] = df['close'].rolling(50, min_periods=1).mean()
    df['ma_200'] = df['close'].rolling(200, min_periods=1).mean()
    
    # 三重过滤：趋势 + 动量 + 超买规避
    df['signal'] = (
        (df['close'] > df['ma_200']) &          # 长期趋势向上
        (df['ma_20'] > df['ma_50']) &           # 短期动量确认
        (df['close'] > df['ma_20'])             # 避免追高（价格回调至MA20以下则离场）
    )
    return df

def trend_following_simple(df, initial_capital=1_000_000):
    """
    🟢 修正版：简单多头策略
    - 信号为 True → 100% 资金买入 BTC
    - 信号为 False → 100% 资金转为现金
    """
    capital = initial_capital
    capital_history = [capital]
    in_position = False
    btc_held = 0.0

    for i in range(len(df)):
        price = df['close'].iloc[i]
        signal = df['signal'].iloc[i] if i >= 50 else False  # 等待指标就绪

        if signal and not in_position:
            # 买入：全部现金换 BTC
            btc_held = capital / price
            in_position = True
        elif not signal and in_position:
            # 卖出：全部 BTC 换现金
            capital = btc_held * price
            btc_held = 0.0
            in_position = False

        # 计算当前资产价值
        current_value = btc_held * price if in_position else capital
        capital_history.append(current_value)

    return np.array(capital_history[:-1])

def buy_and_hold(df, initial_capital=1_000_000):
    returns = df['close'].pct_change().fillna(0)
    return initial_capital * (1 + returns).cumprod()

def original_martingale(df, initial_capital=1_000_000, base_bet=10_000):
    """原始马丁格尔（仅用于对比）"""
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['signal'] = df['return'].shift(1) > 0

    capital = initial_capital
    capital_history = [capital]
    bet = base_bet

    for i in range(len(df) - 1):
        if pd.isna(df['signal'].iloc[i]):
            capital_history.append(capital)
            continue
        fut_ret = df['return'].iloc[i+1]
        pnl = bet * fut_ret if df['signal'].iloc[i] else -bet * fut_ret
        capital += pnl
        capital_history.append(capital)
        bet = min(bet * 2, capital * 0.9) if pnl < 0 else base_bet
        if capital <= 0:
            capital_history.extend([0] * (len(df) - len(capital_history)))
            break
    return np.array(capital_history[:-1])

def max_drawdown(arr):
    arr = np.array(arr)
    peak = np.maximum.accumulate(arr)
    drawdown = (arr - peak) / peak
    return drawdown.min()

def main():
    df = fetch_data()
    df = generate_better_signals(df)
    
    bh = buy_and_hold(df)
    tf = trend_following_simple(df)
    mg = original_martingale(df)
    
    min_len = min(len(bh), len(tf), len(mg))
    bh, tf, mg = bh[:min_len], tf[:min_len], mg[:min_len]
    dates = df['datetime'].iloc[:min_len]
    
    # 绘图
    plt.figure(figsize=(14, 7))
    plt.plot(dates, bh, label='Buy & Hold', linewidth=2, color='black')
    plt.plot(dates, tf, label='Trend Following (Simple)', linewidth=2)
    plt.plot(dates, mg, label='Original Martingale', alpha=0.7)
    plt.title('BTC Strategy Comparison (Fixed Logic)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('strategy_comparison_fixed.png', dpi=150)
    plt.show()
    
    init_cap = 1_000_000
    print(f"\n📈 Final Results (Initial: ${init_cap:,.0f})")
    print(f"Buy & Hold:          ${bh.iloc[-1]:,.0f} ({(bh.iloc[-1]/init_cap - 1)*100:6.2f}%)")
    print(f"Trend Following:     ${tf[-1]:,.0f} ({(tf[-1]/init_cap - 1)*100:6.2f}%)")
    print(f"Original Martingale: ${mg[-1]:,.0f} ({(mg[-1]/init_cap - 1)*100:6.2f}%)")
    
    print(f"\n📉 Max Drawdown")
    print(f"Buy & Hold:          {max_drawdown(bh.values):6.2%}")
    print(f"Trend Following:     {max_drawdown(tf):6.2%}")
    print(f"Original Martingale: {max_drawdown(mg):6.2%}")

if __name__ == "__main__":
    main()