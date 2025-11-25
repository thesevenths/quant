import ccxt
import pandas as pd
from datetime import datetime, timedelta

# 初始化交易所（带代理）
exchange = ccxt.binance({
    'timeout': 15000,
    'enableRateLimit': True,
    'proxies': {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890',
    },
})

def fetch_ohlcv_history(exchange, symbol, timeframe, start_date, end_date=None):
    """
    拉取指定时间范围内的 OHLCV 历史数据（自动分页）
    
    Args:
        exchange: ccxt 交易所实例
        symbol: 交易对，如 'BTC/USDT'
        timeframe: 时间粒度，如 '1h'
        start_date: 起始时间（datetime 对象或 ISO 字符串）
        end_date: 结束时间（可选，默认为当前时间）
    
    Returns:
        pandas.DataFrame: 包含 timestamp, open, high, low, close, volume, datetime
    """
    # 标准化时间为 datetime 对象
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
    if end_date is None:
        end_date = datetime.utcnow()
    elif isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    
    # 转为毫秒时间戳
    since = int(start_date.timestamp() * 1000)
    end_time = int(end_date.timestamp() * 1000)
    
    all_ohlcv = []
    max_limit = 1000  # Binance 最大 limit
    
    while since < end_time:
        try:
            # 拉取一批数据
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=max_limit
            )
            if not ohlcv:
                break
            
            # 过滤掉超出 end_time 的数据（防止最后一批越界）
            ohlcv = [k for k in ohlcv if k[0] <= end_time]
            if not ohlcv:
                break
                
            all_ohlcv.extend(ohlcv)
            
            # 下一批从最后一根K线之后开始
            since = ohlcv[-1][0] + 1
            
            # 防止无限循环（如果交易所返回相同时间戳）
            if len(ohlcv) < max_limit:
                break
                
        except Exception as e:
            print(f"拉取失败 at {since}: {e}")
            break
    
    if not all_ohlcv:
        return pd.DataFrame()
    
    # 去重（按时间戳）
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    return df

# ==============================
# 使用示例：过去 2 年的小时级 BTC/USDT 数据
# ==============================
symbol = 'BTC/USDT'
timeframe = '1h'
start = datetime.utcnow() - timedelta(days=2 * 365)  # 2年前
end = datetime.utcnow()

print("正在拉取过去2年的小时级K线数据...")
df = fetch_ohlcv_history(exchange, symbol, timeframe, start, end)

print(f"共获取 {len(df)} 根K线")
print(df.head())
print(df.tail())


df.to_csv('btc_usdt_2y_1h.csv', index=False)