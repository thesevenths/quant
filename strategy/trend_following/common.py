# common.py
import os
import pandas as pd
import numpy as np

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