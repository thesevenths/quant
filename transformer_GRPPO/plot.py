import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# 加载预测结果
results = pd.read_csv("predictions.csv")

# 删除包含 NaN 的行
results = results.dropna()

# 计算评估指标 
mse = mean_squared_error(results["actual_close"], results["predicted_close"])
mae = mean_absolute_error(results["actual_close"], results["predicted_close"])
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# 绘制预测值与实际值对比图
plt.figure(figsize=(12, 6))
plt.plot(results.index, results["actual_close"], label="Actual Close Price", color="blue")
plt.plot(results.index, results["predicted_close"], label="Predicted Close Price", color="red", linestyle="--")
plt.xlabel("Sample Index")
plt.ylabel("Close Price")
plt.title("Bitcoin Close Price: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.savefig("price_prediction_plot.png")
plt.show()
print("Plot saved to price_prediction_plot.png")