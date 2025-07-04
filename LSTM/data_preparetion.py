import pandas as pd
import numpy as np
import os

# Set data directory
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)
os.makedirs(os.path.join(data_dir, 'calendars'), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'instruments'), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'features', 'BTC'), exist_ok=True)

# Load CSV data
df = pd.read_csv('/btc_daily.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')

# Extract calendar (dates in YYYYMMDD format)
calendar = df['datetime'].dt.strftime('%Y%m%d').unique()
with open(os.path.join(data_dir, 'calendars', 'day.txt'), 'w') as f:
    for date in calendar:
        f.write(date + '\n')

# Instruments file
start_date = df['datetime'].min().strftime('%Y%m%d')
end_date = df['datetime'].max().strftime('%Y%m%d')
with open(os.path.join(data_dir, 'instruments', 'all.txt'), 'w') as f:
    f.write(f'BTC {start_date} {end_date}\n')

# Save features as binary files
features = ['open', 'high', 'low', 'close', 'volume']
for feature in features:
    values = df[feature].values.astype(np.float32)
    values.tofile(os.path.join(data_dir, 'features', 'BTC', f'{feature}.bin'))