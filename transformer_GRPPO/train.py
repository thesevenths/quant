import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Ensure Keras 2 compatibility
os.environ["TRL_PROFILE"] = "0"  # 禁用 trl 的 profiling 功能
os.environ["MLFLOW_TRACKING_URI"] = ""  # 禁用 mlflow 日志


import sys
# 切换到当前脚本所在目录
os.chdir(sys.path[0])
print("Current working directory:", os.getcwd())

import re
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
import numpy as np

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 模型路径
model_path = "F:\\AI_Quant\\model\\Qwen\\Qwen3-0.6B"
checkpoint_dir = "F:\\AI_Quant\\model\\Qwen"
os.makedirs(checkpoint_dir, exist_ok=True)

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
).to(device)

# 配置 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.generation_config.enable_thinking = True

# 加载数据
data_path = "btc_daily.csv"
df = pd.read_csv(data_path)
features = df[['open', 'low', 'high', 'volume']].values
targets = df['close'].values


# 格式化输入
def format_input(open_price, low_price, high_price, volume):
    return f'''
        ## Role
        - A world-class trading expert in finance market
        
        ## Skills
        - a profound knowledge of the history and future prospects of crypto
        - A top economist with a deep understanding of global economic trends
        
        ## question
        - Predict the close price based on: open: {open_price}, low: {low_price}, high: {high_price}, volume: {volume}
        
        ## Constrains
        - Strictly follow the user's input and output requirements
        - inference/reasoning step by step

        ## OutputFormat
        - - <think>....</think> [digital number]
        - between the 2 <think> .... </think> label, show users your reasoning process, which indicate how to predict the close price
        - after the </think> label is the final answer, which should be only one number , no other characters, so that i can easily extract the close price
    '''


# 创建数据集
data = {
    "prompt": [format_input(f[0], f[1], f[2], f[3]) for f in features],
    "answer": [str(t) for t in targets]
}
dataset = Dataset.from_dict(data)


def reward_function(prompts, completions, answer, **kwargs) -> list[float]:
    rewards = []
    for prompt, completion, ground_truth in zip(prompts, completions, answer):
        try:
            # 调试输出
            print("reward_function prompt:", prompt)
            print("reward_function completion:", completion)
            print("reward_function ground_truth:", ground_truth)
            # 提取生成文本
            if isinstance(completion, dict):
                content_str = completion.get('generated_text', completion.get('content', '')).strip()
            elif isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                content_str = completion[0].get('generated_text', completion[0].get('content', '')).strip()
            else:
                content_str = str(completion).strip()

            # 检查整体格式：<think>...</think>content
            if not ("<think>" in content_str and "</think>" in content_str):
                rewards.append(-10.0)  # 格式不符合，奖励 -10
                continue

            # 提取 content（</think> 后的部分）
            content_start = content_str.index("</think>") + len("</think>")
            content_text = content_str[content_start:].strip()

            # 惩罚冗余输出  
            content_length_penalty = -len(content_text) * 0.001  # 每字符扣 0.001 分
            if len(content_text) > 20:  # 假设价格数值不超过 20 字符
                rewards.append(-5.0 + content_length_penalty)  # 过长输出，奖励 -5 + 长度惩罚
                continue

            # 检查 content 是否为纯数字（允许小数点和负号）
            if not re.fullmatch(r'[-+]?\d*\.?\d+', content_text):
                rewards.append(-5.0 + content_length_penalty)  # 非数值 token，奖励 -5 + 长度惩罚
                continue

            # 转换为浮点数并计算奖励
            predicted = float(content_text)
            if not np.isfinite(predicted):
                rewards.append(-5.0 + content_length_penalty)  # 非有限数值，奖励 -5 + 长度惩罚
                continue
            print("reward_function predicted answer:", predicted)
            print("===================================================================================")
            actual = float(ground_truth)
            error = abs(predicted - actual) / actual
            if error <= 0.001:  # 误差 ≤ 0.1%
                reward = 10.0
            else:
                reward = -abs(predicted - actual)
            rewards.append(reward + content_length_penalty)

        except (ValueError, IndexError, KeyError) as e:
            print(f"Error in reward_function: {e}")
            rewards.append(-10.0)  # 整体格式错误，奖励 -10

    return rewards

grpo_config = GRPOConfig(
    output_dir=checkpoint_dir,
    per_device_train_batch_size=4,
    num_generations=4,
    max_steps=200,
    learning_rate=1e-5,
    save_strategy="epoch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_grad_norm=0.1,
    save_steps=1,
    logging_steps=10,
    gradient_accumulation_steps=4,
    bf16=torch.cuda.is_available(),  # 启用 bf16 如果 GPU 可用
    fp16=False,
    tf32=False,
    log_on_each_node=False,
    report_to="none",  # 禁用所有外部日志（wandb, mlflow 等）
    disable_tqdm=False  # 启用进度条，便于调试
)

# 初始化 GRPOTrainer
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    reward_funcs=reward_function
)

# 训练
print("Starting GRPO training...")
trainer.train()

# 保存最终模型
final_checkpoint = os.path.join(checkpoint_dir, "final_model")
model.save_pretrained(final_checkpoint)
tokenizer.save_pretrained(final_checkpoint)
print(f"Final model saved to {final_checkpoint}")
