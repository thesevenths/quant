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
    return df.sort_values('datetime').reset_index(drop=True)

def strategy_backtest(df, initial_capital=1_000_000, buy_percent=0.01, threshold=0.01, mode='down'):
    """
    回测策略：
    - mode='down': 下跌加仓（close/open - 1 <= -threshold）
    - mode='up': 上涨加仓（close/open - 1 >= threshold）
    """
    cash = initial_capital
    btc_balance = 0.0
    portfolio_values = []

    # 遍历每一根K线（T时刻）
    for i in range(len(df)):
        price = df['close'].iloc[i]
        total_value = cash + btc_balance * price
        portfolio_values.append(total_value)

        # 决定 T+1 是否买入（但不能超出数据范围）
        if i + 1 >= len(df):
            break

        open_price = df['open'].iloc[i]
        close_price = df['close'].iloc[i]
        ret = (close_price - open_price) / open_price

        should_buy = False
        if mode == 'down' and ret <= -threshold:
            should_buy = True
        elif mode == 'up' and ret >= threshold:
            should_buy = True

        if should_buy:
            # 在 T+1 时刻买入：使用 T+1 的 open 价格（更真实）
            next_open = df['open'].iloc[i + 1]
            total_value_before_buy = cash + btc_balance * next_open
            invest_amount = total_value_before_buy * buy_percent
            if invest_amount > cash:
                invest_amount = cash  # 防止现金不足
            
            btc_bought = invest_amount / next_open
            btc_balance += btc_bought
            cash -= invest_amount

    return np.array(portfolio_values)

def buy_and_hold(df, initial_capital=1_000_000):
    initial_price = df['open'].iloc[0]  # 假设在第一个小时开盘买入
    btc_bought = initial_capital / initial_price
    return btc_bought * df['close'].values

def main():
    df = fetch_data()
    
    # 回测三种策略
    down_strategy = strategy_backtest(df, mode='down', buy_percent=0.01, threshold=0.01)
    up_strategy = strategy_backtest(df, mode='up', buy_percent=0.01, threshold=0.01)
    bh_values = buy_and_hold(df)

    # 对齐长度
    min_len = min(len(down_strategy), len(up_strategy), len(bh_values))
    down_strategy = down_strategy[:min_len]
    up_strategy = up_strategy[:min_len]
    bh_values = bh_values[:min_len]
    df_plot = df.iloc[:min_len].copy()

    # 绘图
    plt.figure(figsize=(14, 7))
    plt.plot(df_plot['datetime'], down_strategy, label='Buy the dip (fall >1%, invest 1%)', linewidth=1.5)
    plt.plot(df_plot['datetime'], up_strategy, label='Add on declines (drop >1%, buy 1%)', linewidth=1.5)
    plt.plot(df_plot['datetime'], bh_values, label='Buy and Hold', linewidth=1.5, alpha=0.7)
    plt.title('BTC 小时级加仓策略回测（2年）')
    plt.xlabel('时间')
    plt.ylabel('账户总值（USDT）')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('martingale_like_strategies.png')
    plt.show()

    # 输出最终结果
    initial = 1_000_000
    """
        下跌加仓最终: 1,634,426 (63.44%)
        上涨加仓最终: 1,781,547 (78.15%)
        买入持有最终: 2,335,500 (133.55%)
        还不如buy and hold……
    """
    print(f"初始资金: {initial:,.0f} USDT")
    print(f"下跌加仓最终: {down_strategy[-1]:,.0f} ({(down_strategy[-1]/initial - 1)*100:.2f}%)")
    print(f"上涨加仓最终: {up_strategy[-1]:,.0f} ({(up_strategy[-1]/initial - 1)*100:.2f}%)")
    print(f"买入持有最终: {bh_values[-1]:,.0f} ({(bh_values[-1]/initial - 1)*100:.2f}%)")

    print("\n" + "="*60)
    print("📌 策略说明：")
    print("- 下跌加仓：当某小时跌幅 ≥1%（close/open），下个小时用总资产1%买入")
    print("- 上涨加仓：当某小时涨幅 ≥1%，下个小时用总资产1%买入")
    print("- 所有买入按下一小时开盘价成交，无手续费，无滑点（理想化）")
    print("- 现金永不为负，买入时最多用光现金")
    print("="*60)

if __name__ == "__main__":
    main()