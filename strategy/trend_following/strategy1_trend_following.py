# strategy1_trend_following.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from common import fetch_data, buy_and_hold, max_drawdown

def trend_following_strategy(df, initial_capital=1_000_000, ma_window=50):
    df = df.copy()
    df['ma'] = df['close'].rolling(ma_window).mean()
    
    cash = initial_capital
    btc = 0.0
    in_position = False
    portfolio = []
    
    for i, row in df.iterrows():
        price = row['close']
        total = cash + btc * price
        portfolio.append(total)
        
        if i < ma_window:
            continue
        
        if row['close'] > row['ma']:
            if not in_position:
                # 全仓买入（下一小时开盘价）
                if i + 1 < len(df):
                    buy_price = df['open'].iloc[i + 1]
                    btc = cash / buy_price
                    cash = 0
                    in_position = True
        else:
            if in_position:
                # 全部卖出（下一小时开盘价）
                if i + 1 < len(df):
                    sell_price = df['open'].iloc[i + 1]
                    cash = btc * sell_price
                    btc = 0
                    in_position = False
    
    return np.array(portfolio)

def main():
    df = fetch_data()
    strategy1 = trend_following_strategy(df, ma_window=50)
    bh = buy_and_hold(df)
    
    min_len = min(len(strategy1), len(bh))
    strategy1 = strategy1[:min_len]
    bh = bh[:min_len]
    dates = df['datetime'].iloc[:min_len]
    
    plt.figure(figsize=(14, 7))
    plt.plot(dates, strategy1, label='趋势跟踪（MA50）', linewidth=1.5)
    plt.plot(dates, bh, label='买入持有', alpha=0.7)
    plt.title('方案1：趋势跟踪策略')
    plt.legend()
    plt.grid(True)
    plt.savefig('strategy1_trend.png')
    plt.show()
    """
    === 方案1：趋势跟踪 ===
    最终收益: 1,791,344 (79.13%)
    最大回撤: 29.03%
    买入持有: 2,335,500 (34.76% 回撤)
    """
    print("=== 方案1：趋势跟踪 ===")
    print(f"最终收益: {strategy1[-1]:,.0f} ({(strategy1[-1]/1e6 - 1)*100:.2f}%)")
    print(f"最大回撤: {max_drawdown(strategy1)*100:.2f}%")
    print(f"买入持有: {bh[-1]:,.0f} ({max_drawdown(bh)*100:.2f}% 回撤)")

if __name__ == "__main__":
    main()