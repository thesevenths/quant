# strategy3_valuation_signal.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from common import fetch_data, buy_and_hold, max_drawdown

def valuation_based_strategy(df, initial_capital=1_000_000):
    df = df.copy()
    df['ma200'] = df['close'].rolling(200 * 24).mean()  # 200天 = 4800小时
    df['z_score'] = (df['close'] - df['ma200']) / df['close'].rolling(200 * 24).std()
    
    cash = initial_capital
    btc = 0.0
    in_position = False
    portfolio = []
    
    for i, row in df.iterrows():
        price = row['close']
        total = cash + btc * price
        portfolio.append(total)
        
        if i < 200 * 24:
            continue
        
        # 高估区域（Z > 2）→ 空仓；否则持仓
        if row['z_score'] > 2.0:
            if in_position:
                if i + 1 < len(df):
                    sell_price = df['open'].iloc[i + 1]
                    cash = btc * sell_price
                    btc = 0
                    in_position = False
        else:
            if not in_position:
                if i + 1 < len(df):
                    buy_price = df['open'].iloc[i + 1]
                    btc = cash / buy_price
                    cash = 0
                    in_position = True
    
    return np.array(portfolio)

def main():
    df = fetch_data()
    strategy3 = valuation_based_strategy(df)
    bh = buy_and_hold(df)
    
    min_len = min(len(strategy3), len(bh))
    strategy3 = strategy3[:min_len]
    bh = bh[:min_len]
    dates = df['datetime'].iloc[:min_len]
    
    plt.figure(figsize=(14, 7))
    plt.plot(dates, strategy3, label='估值信号策略', linewidth=1.5)
    plt.plot(dates, bh, label='买入持有', alpha=0.7)
    plt.title('方案3：宏观估值信号')
    plt.legend()
    plt.grid(True)
    plt.savefig('strategy3_valuation.png')
    plt.show()
    
    """
    === 方案3：估值信号策略 ===
    最终收益: 1,048,564 (4.86%)
    最大回撤: 34.76%
    买入持有: 2,335,500 (34.76% 回撤)
💡 逻辑：价格显著高于200日均线时减仓
    """
    print("=== 方案3：估值信号策略 ===")
    print(f"最终收益: {strategy3[-1]:,.0f} ({(strategy3[-1]/1e6 - 1)*100:.2f}%)")
    print(f"最大回撤: {max_drawdown(strategy3)*100:.2f}%")
    print(f"买入持有: {bh[-1]:,.0f} ({max_drawdown(bh)*100:.2f}% 回撤)")
    print("💡 逻辑：价格显著高于200日均线时减仓")

if __name__ == "__main__":
    main()