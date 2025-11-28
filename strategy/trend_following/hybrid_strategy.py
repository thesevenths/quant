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

def buy_and_hold(df, initial_capital=1_000_000):
    initial_price = df['open'].iloc[0]
    btc_bought = initial_capital / initial_price
    return btc_bought * df['close'].values

def max_drawdown(values):
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    return np.max(drawdown)

def hybrid_strategy(
    df,
    initial_capital=1_000_000,
    ma_window=50,
    vol_window=24,
    target_vol=0.015,      # 目标小时波动率 1.5%
    max_leverage=1.0,
    extreme_vol_threshold=0.05,  # 单小时涨跌 >5% 视为极端
    extreme_vol_max_weight=0.5   # 极端波动时最大仓位50%
):
    df = df.copy()
    # 计算指标
    df['ma'] = df['close'].rolling(ma_window * 24).mean()  # MA50天 = 1200小时
    df['hourly_return'] = df['close'].pct_change()
    df['vol'] = df['hourly_return'].rolling(vol_window).std()
    
    cash = initial_capital
    btc = 0.0
    portfolio = []
    last_rebalance_day = None
    
    for i in range(len(df)):
        row = df.iloc[i]
        price = row['close']
        total_value = cash + btc * price
        portfolio.append(total_value)
        
        # 每周调仓（每周一 00:00）
        current_day = row['datetime'].date()
        is_monday = row['datetime'].weekday() == 0
        is_midnight = row['datetime'].hour == 0
        
        should_rebalance = False
        if last_rebalance_day is None:
            should_rebalance = True
        elif (current_day - last_rebalance_day).days >= 7:
            should_rebalance = True
        elif is_monday and is_midnight and current_day != last_rebalance_day:
            should_rebalance = True
        
        if not should_rebalance or i < ma_window * 24:
            continue
        
        # 趋势判断
        in_uptrend = row['close'] > row['ma']
        
        if not in_uptrend:
            # 熊市：空仓
            if btc > 0:
                cash += btc * price
                btc = 0
            last_rebalance_day = current_day
            continue
        
        # 牛市：计算波动率仓位
        current_vol = df['vol'].iloc[i]
        if pd.isna(current_vol) or current_vol == 0:
            target_weight = max_leverage
        else:
            target_weight = target_vol / current_vol
            target_weight = min(target_weight, max_leverage)
            target_weight = max(target_weight, 0.0)
        
        # 极端波动保护
        if abs(row['hourly_return']) > extreme_vol_threshold:
            target_weight = min(target_weight, extreme_vol_max_weight)
        
        # 调整仓位
        target_btc_value = total_value * target_weight
        target_btc = target_btc_value / price
        delta_btc = target_btc - btc
        
        if delta_btc > 0:
            cost = delta_btc * price
            if cost <= cash:
                btc += delta_btc
                cash -= cost
        elif delta_btc < 0:
            proceeds = -delta_btc * price
            btc += delta_btc
            cash += proceeds
        
        last_rebalance_day = current_day
    
    return np.array(portfolio)

def main():
    df = fetch_data()
    
    # 回测混合策略
    hybrid_vals = hybrid_strategy(df)
    bh_vals = buy_and_hold(df)
    
    # 对齐长度
    min_len = min(len(hybrid_vals), len(bh_vals))
    hybrid_vals = hybrid_vals[:min_len]
    bh_vals = bh_vals[:min_len]
    dates = df['datetime'].iloc[:min_len]
    
    # 绘图
    plt.figure(figsize=(14, 7))
    plt.plot(dates, hybrid_vals, label='混合策略（趋势+波动率）', linewidth=1.5)
    plt.plot(dates, bh_vals, label='买入持有', alpha=0.7)
    plt.title('混合策略 vs 买入持有（目标：回撤 <30%）')
    plt.xlabel('时间')
    plt.ylabel('账户总值（USDT）')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('hybrid_strategy.png')
    plt.show()
    
    # 结果
    initial = 1_000_000
    hybrid_return = (hybrid_vals[-1] / initial - 1) * 100
    bh_return = (bh_vals[-1] / initial - 1) * 100
    hybrid_dd = max_drawdown(hybrid_vals) * 100
    bh_dd = max_drawdown(bh_vals) * 100
    
    """
    ============================================================
    📊 混合策略回测结果
    初始资金: 1,000,000 USDT
    混合策略最终: 1,280,030 (28.00%)
    买入持有最终: 2,335,500 (133.55%)
    混合策略最大回撤: 41.62%
    买入持有最大回撤: 34.76%

    ⚠️ 回撤未控制在30%以内
    ============================================================
    """
    print("="*60)
    print("📊 混合策略回测结果")
    print(f"初始资金: {initial:,.0f} USDT")
    print(f"混合策略最终: {hybrid_vals[-1]:,.0f} ({hybrid_return:.2f}%)")
    print(f"买入持有最终: {bh_vals[-1]:,.0f} ({bh_return:.2f}%)")
    print(f"混合策略最大回撤: {hybrid_dd:.2f}%")
    print(f"买入持有最大回撤: {bh_dd:.2f}%")
    print()
    if hybrid_dd < 30:
        print("✅ 成功实现：回撤 < 30%！")
    else:
        print("⚠️ 回撤未控制在30%以内")
    print("="*60)

if __name__ == "__main__":
    main()