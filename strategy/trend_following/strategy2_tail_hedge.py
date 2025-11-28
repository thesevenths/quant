# strategy2_tail_hedge.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from common import fetch_data, buy_and_hold, max_drawdown

def tail_risk_hedging_strategy(df, initial_capital=1_000_000, hedge_cost_rate=0.02, crash_threshold=-0.20, hedge_payout=5.0):
    """
    模拟期权对冲：
    - 每年从收益中扣除 hedge_cost_rate 作为保险费
    - 当单日跌幅 >= crash_threshold，获得 hedge_payout 倍保险金额赔付
    """
    df = df.copy()
    df['daily_return'] = df['close'].pct_change(periods=24)  # 24小时收益
    
    cash = initial_capital
    btc = cash / df['open'].iloc[0]  # 初始全仓买入
    cash = 0
    
    portfolio = []
    last_hedge_time = df['datetime'].iloc[0]
    total_hedge_premium = 0
    
    for i, row in df.iterrows():
        price = row['close']
        total = cash + btc * price
        portfolio.append(total)
        
        # 每年支付保险费（从总值中扣除）
        if (row['datetime'] - last_hedge_time).days >= 365:
            hedge_premium = total * hedge_cost_rate
            total_hedge_premium += hedge_premium
            # 从现金中扣除（若无现金，卖出 BTC）
            if cash >= hedge_premium:
                cash -= hedge_premium
            else:
                # 卖出部分 BTC 支付保费
                need_sell = (hedge_premium - cash) / price
                btc -= need_sell
                cash = 0
            last_hedge_time = row['datetime']
        
        # 检查是否发生暴跌（过去24小时）
        if i >= 24 and df['daily_return'].iloc[i] <= crash_threshold:
            # 触发保险赔付：赔付 = 保费累计 × payout
            payout = total_hedge_premium * hedge_payout
            cash += payout
            total_hedge_premium = 0  # 重置（单次赔付）
    
    return np.array(portfolio)

def main():
    df = fetch_data()
    strategy2 = tail_risk_hedging_strategy(df)
    bh = buy_and_hold(df)
    
    min_len = min(len(strategy2), len(bh))
    strategy2 = strategy2[:min_len]
    bh = bh[:min_len]
    dates = df['datetime'].iloc[:min_len]
    
    plt.figure(figsize=(14, 7))
    plt.plot(dates, strategy2, label='尾部对冲（模拟期权）', linewidth=1.5)
    plt.plot(dates, bh, label='买入持有', alpha=0.7)
    plt.title('方案2：尾部风险对冲')
    plt.legend()
    plt.grid(True)
    plt.savefig('strategy2_hedge.png')
    plt.show()
    """
    === 方案2：尾部风险对冲 ===
    最终收益: 2,288,790 (128.88%)
    最大回撤: 34.76%
    买入持有: 2,335,500 (34.76% 回撤)
    模拟：每年支付 2% 保费，暴跌时获得 5 倍赔付
    """
    print("=== 方案2：尾部风险对冲 ===")
    print(f"最终收益: {strategy2[-1]:,.0f} ({(strategy2[-1]/1e6 - 1)*100:.2f}%)")
    print(f"最大回撤: {max_drawdown(strategy2)*100:.2f}%")
    print(f"买入持有: {bh[-1]:,.0f} ({max_drawdown(bh)*100:.2f}% 回撤)")
    print("💡 模拟：每年支付 2% 保费，暴跌时获得 5 倍赔付")

if __name__ == "__main__":
    main()