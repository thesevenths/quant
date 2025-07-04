import ccxt
import pandas as pd
import requests


def acquire_data(save_path):
    exchange = ccxt.okx({
        'enableRateLimit': True,
        'proxies': {
            'http': 'http://192.168.50.45:8181',
            'https': 'http://192.168.50.45:8181',
        },
    })

    since = exchange.parse8601('2022-01-01T00:00:00Z')
    bars = exchange.fetch_ohlcv('BTC/USDT', '1d', since)
    df = pd.DataFrame(bars, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df['symbol'] = 'BTC'
    df.to_csv(save_path, index=False)
    print("数据已保存到：", save_path)


if __name__ == '__main__':
    # acquire_data('btc_daily_okx.csv')
    import qlib
    print(qlib.__version__)