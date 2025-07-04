import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 模型路径
checkpoint_path = "E:\\AI_Quant\\model\\Qwen\\final_model"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    torch_dtype="auto",
    device_map="auto"
).to(device)

# 加载数据 
data_path = "btc_daily.csv"
df = pd.read_csv(data_path)
features = df[['open', 'low', 'high', 'volume']].values
targets = df['close'].values

# 格式化输入
def format_input(open_price, low_price, high_price, volume):
    return f"Predict the close price based on: open: {open_price}, low: {low_price}, high: {high_price}, volume: {volume}"

# 预测
# predictions = []
# model.eval()
# with torch.no_grad():
#     for f in features:
#         prompt = format_input(f[0], f[1], f[2], f[3])
#         inputs = tokenizer(prompt, return_tensors="pt").to(device)
#         outputs = model.generate(**inputs, max_new_tokens=32)
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         try:
#             pred_value = float(response.split(prompt)[-1].strip())
#         except ValueError:
#             pred_value = np.nan  # 处理无效输出
#         predictions.append(pred_value)

# 预测
predictions = []
model.eval()
with torch.no_grad():
    for f in features:
        prompt = format_input(f[0], f[1], f[2], f[3])
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, enable_thinking=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=32)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        try:
            # 解析 </think> 后的 content
            index = response.index("</think>") if "</think>" in response else 0
            content = response[index + len("</think>"):].strip() if index > 0 else response.strip()
            pred_value = float(content)
            if not np.isfinite(pred_value) or content.strip() != str(pred_value).strip():
                raise ValueError("Invalid content format")
        except (ValueError, IndexError):
            pred_value = np.nan  # 无效格式设为 NaN
        predictions.append(pred_value)


# 保存预测结果
results = pd.DataFrame({
    "open": features[:, 0],
    "low": features[:, 1],
    "high": features[:, 2],
    "volume": features[:, 3],
    "actual_close": targets,
    "predicted_close": predictions
})
results.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")