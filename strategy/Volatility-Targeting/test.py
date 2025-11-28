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

def volatility_targeting_strategy(
    df,
    initial_capital=1_000_000,
    target_vol=0.02,        # 目标波动率（每小时）
    vol_window=24,          # 波动率计算窗口（小时）
    max_leverage=1.0        # 最大杠杆（1.0 = 不加杠杆）
):
    """
    波动率自适应仓位策略：
    - 用过去 vol_window 小时的收益率标准差估计波动率
    - 仓位比例 = min(target_vol / realized_vol, max_leverage)
    - 每小时调整一次仓位（理想化，无手续费）
    """
    df = df.copy()
    
    # 计算每小时收益率
    df['hourly_return'] = df['close'].pct_change()
    
    # 计算滚动波动率（标准差）
    df['vol'] = df['hourly_return'].rolling(window=vol_window).std()
    
    cash = initial_capital
    btc_balance = 0.0
    portfolio_values = []
    
    for i in range(len(df)):
        price = df['close'].iloc[i]
        total_value = cash + btc_balance * price
        portfolio_values.append(total_value)
        
        if i < vol_window:  # 波动率未计算完成
            continue
        
        current_vol = df['vol'].iloc[i]
        if pd.isna(current_vol) or current_vol == 0:
            target_weight = 0.0
        else:
            # 核心：仓位 = 目标波动率 / 当前波动率
            target_weight = target_vol / current_vol
            # 限制最大仓位（防止单日波动过低导致满仓）
            target_weight = min(target_weight, max_leverage)
            target_weight = max(target_weight, 0.0)  # 不做空
        
        # 目标 BTC 价值 = total_value * target_weight
        target_btc_value = total_value * target_weight
        target_btc = target_btc_value / price
        
        # 调整仓位（买入或卖出）
        delta_btc = target_btc - btc_balance
        if delta_btc > 0:
            # 买入
            cost = delta_btc * price
            if cost <= cash:
                btc_balance += delta_btc
                cash -= cost
        elif delta_btc < 0:
            # 卖出
            proceeds = -delta_btc * price
            btc_balance += delta_btc  # delta_btc 为负
            cash += proceeds
    
    return np.array(portfolio_values)

def buy_and_hold(df, initial_capital=1_000_000):
    initial_price = df['open'].iloc[0]
    btc_bought = initial_capital / initial_price
    return btc_bought * df['close'].values

def max_drawdown(values):
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    return np.max(drawdown)

def main():
    df = fetch_data()
    
    # 回测波动率策略
    vol_strategy = volatility_targeting_strategy(
        df,
        initial_capital=1_000_000,
        target_vol=0.01,     # 目标每小时波动率 1%
        vol_window=24,       # 用过去24小时估算波动率
        max_leverage=1.0     # 最多100%仓位
    )
    bh_values = buy_and_hold(df)
    
    # 对齐
    min_len = min(len(vol_strategy), len(bh_values))
    vol_strategy = vol_strategy[:min_len]
    bh_values = bh_values[:min_len]
    df_plot = df.iloc[:min_len].copy()
    
    # 绘图
    plt.figure(figsize=(14, 7))
    plt.plot(df_plot['datetime'], vol_strategy, label='波动率自适应策略', linewidth=1.5)
    plt.plot(df_plot['datetime'], bh_values, label='买入持有', linewidth=1.5, alpha=0.7)
    plt.title('波动率自适应仓位 vs 买入持有')
    plt.xlabel('时间')
    plt.ylabel('账户总值（USDT）')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('volatility_targeting.png')
    plt.show()
    
    # 输出结果
    initial = 1_000_000
    print(f"初始资金: {initial:,.0f}")
    print(f"波动率策略最终: {vol_strategy[-1]:,.0f} ({(vol_strategy[-1]/initial - 1)*100:.2f}%)")
    print(f"买入持有最终: {bh_values[-1]:,.0f} ({(bh_values[-1]/initial - 1)*100:.2f}%)")
    print(f"波动率策略最大回撤: {max_drawdown(vol_strategy)*100:.2f}%")
    print(f"买入持有最大回撤: {max_drawdown(bh_values)*100:.2f}%")
    
    print("\n" + "="*60)
    print("📌 策略关键点：")
    print("1. 仓位 = min(目标波动率 / 实现波动率, 最大杠杆)")
    print("2. 高波动时自动减仓，低波动时加仓")
    print("3. 不预测方向，只管理风险")
    print("4. 在剧烈下跌中，仓位已降低，回撤更小")
    print("5. 代价：在单边牛市中，早期仓位不足，收益低于 Buy & Hold")
    print("="*60)

if __name__ == "__main__":
    main()