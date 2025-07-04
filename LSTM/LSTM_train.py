import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

# 设置随机种子以保证结果可复现
np.random.seed(42)


# 1. 数据加载与预处理
def load_data(file_path='E:/AI_Quant/data/btc_daily.csv'):
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    data = df['close'].values.reshape(-1, 1)
    return data, df.index


# 2. 创建时间序列数据集
def create_dataset(data, look_back=60):
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)


# 3. 划分训练/验证/测试集
def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * val_ratio)
    test_size = len(X) - train_size - val_size

    X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test


# 4. 构建LSTM模型
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# 5. 可视化预测结果
def plot_predictions(data, train_predict, val_predict, test_predict, train_size, val_size):
    plt.figure(figsize=(14, 6))
    plt.plot(data, label='Actual Price')

    # 绘制训练集预测
    plt.plot(range(train_size), train_predict[:, 0], label='Train Predict')
    # 绘制验证集预测
    plt.plot(range(train_size, train_size + val_size), val_predict[:, 0], label='Validation Predict')
    # 绘制测试集预测
    plt.plot(range(train_size + val_size, len(data)), test_predict[:, 0], label='Test Predict')

    plt.legend()
    plt.title('BTC Closing Price Prediction with LSTM')
    plt.xlabel('Time (days)')
    plt.ylabel('Price (USD)')
    plt.show()


# 6. 累计收益计算与可视化
def calculate_cumulative_return(predictions, actuals):
    returns = np.diff(actuals) / actuals[:-1]
    pred_returns = np.diff(predictions) / predictions[:-1]

    # 对齐长度
    L = min(len(returns), len(pred_returns))
    returns = returns[:L]
    pred_returns = pred_returns[:L]

    strategy_returns = np.where(pred_returns > 0, returns, -returns)
    return np.cumprod(1 + strategy_returns)


def plot_cumulative_return(train_actual, train_pred, val_actual, val_pred, test_actual, test_pred):
    # 确保预测值和实际值的长度一致
    train_cr = calculate_cumulative_return(train_pred[:, 0], train_actual)
    val_cr = calculate_cumulative_return(val_pred[:, 0], val_actual)
    test_cr = calculate_cumulative_return(test_pred[:, 0], test_actual)

    full_cr = np.concatenate((train_cr, val_cr, test_cr))

    plt.figure(figsize=(14, 6))
    plt.plot(full_cr, label='Cumulative Return')
    plt.axvline(len(train_cr) - 1, color='r', linestyle='--', label='Train/Val Split')
    plt.axvline(len(train_cr) + len(val_cr) - 1, color='g', linestyle='--', label='Val/Test Split')
    plt.legend()
    plt.title('BTC Trading Cumulative Return')
    plt.xlabel('Time (days)')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.show()


def build_prediction_dataframe(dates, y, train_predict, val_predict, test_predict, look_back, train_size, val_size):
    """
    构建包含时间、真实价格、预测价格、策略收益率的DataFrame
    """
    total_len = len(train_predict) + len(val_predict) + len(test_predict)
    start_idx = look_back + 1  # 对应第一个可用预测值的位置

    # 对应的时间索引
    valid_dates = dates[start_idx:start_idx + total_len]

    # 拼接预测值
    full_predict = np.concatenate([train_predict, val_predict, test_predict]).flatten()

    # 反归一化真实值
    actual_prices = scaler.inverse_transform(y.reshape(-1, 1)).flatten()

    actual_prices = actual_prices[:total_len]

    # 计算策略收益率：如果预测涨就做多，否则做空
    actual_returns = np.diff(actual_prices) / actual_prices[:-1]
    pred_returns = np.diff(full_predict) / full_predict[:-1]
    strategy_returns = np.where(pred_returns > 0, actual_returns, -actual_returns)
    strategy_returns = np.insert(strategy_returns, 0, 0)  # 补一个0开头，长度一致

    df = pd.DataFrame({
        'datetime': valid_dates,
        'actual_price': actual_prices,
        'predicted_price': full_predict,
        'strategy_return': strategy_returns
    })
    df.set_index('datetime', inplace=True)
    return df


def plot_price_and_return(df):
    """
    绘制实际价格、预测价格与策略收益率（共享时间轴）
    """
    fig, ax1 = plt.subplots(figsize=(15, 6))

    # 价格
    ax1.plot(df.index, df['actual_price'], label='Actual Price', color='black', linewidth=1.5)
    ax1.plot(df.index, df['predicted_price'], label='Predicted Price', color='blue', linestyle='--')
    ax1.set_ylabel('Price (USD)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')

    # 策略收益率（右轴）
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['strategy_return'], label='Strategy Return', color='green', alpha=0.4)
    ax2.set_ylabel('Strategy Return', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')

    plt.title('BTC Actual vs Predicted Price and Strategy Return')
    plt.grid(True)
    plt.show()


# 主程序入口
if __name__ == '__main__':
    MODEL_PATH = 'lstm_btc_model.h5'

    # 加载原始数据
    data, dates = load_data()

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # 构造时间序列数据
    look_back = 60
    X, y = create_dataset(scaled_data, look_back=look_back)

    # 划分数据集
    X = X.reshape((X.shape[0], X.shape[1], 1))  # reshape为LSTM输入格式
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    # 检查模型是否存在
    if os.path.exists(MODEL_PATH):
        print("✅ 模型已存在，正在加载...")
        model = load_model(MODEL_PATH)
    else:
        print("❌ 模型不存在，开始训练...")
        model = build_lstm_model((X_train.shape[1], 1))
        history = model.fit(X_train, y_train,
                            epochs=50,
                            batch_size=32,
                            validation_data=(X_val, y_val),
                            verbose=1)
        model.save(MODEL_PATH)
        print(f"✅ 模型训练完成并已保存至 {MODEL_PATH}")

    # 预测
    train_predict = model.predict(X_train)
    val_predict = model.predict(X_val)
    test_predict = model.predict(X_test)

    # 反归一化
    train_predict = scaler.inverse_transform(train_predict)
    val_predict = scaler.inverse_transform(val_predict)
    test_predict = scaler.inverse_transform(test_predict)
    actual_prices = scaler.inverse_transform([y])  # 反归一化真实值

    # 可视化预测结果
    plot_predictions(actual_prices[0], train_predict, val_predict, test_predict, len(train_predict), len(val_predict))

    # 可视化累计收益
    plot_cumulative_return(
        scaler.inverse_transform([y_train])[0],
        train_predict,
        scaler.inverse_transform([y_val])[0],
        val_predict,
        scaler.inverse_transform([y_test])[0],
        test_predict
    )

    # 构建 DataFrame 并可视化
    df_plot = build_prediction_dataframe(dates, y, train_predict, val_predict, test_predict, look_back,
                                         len(train_predict), len(val_predict))
    plot_price_and_return(df_plot)

    # 如果你想保存为 CSV 可加上：
    df_plot.to_csv('btc_lstm_predictions_with_returns.csv')
