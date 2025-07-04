import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 模型路径
model_name = "E:\\AI_Quant\\model\\Qwen\\Qwen3-0.6B"
print(f"Loading model from: {model_name}")

# 加载 tokenizer 和模型，并将其移动到指定设备
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
).to(device)

# 启动交互式问答循环
print("\n开始对话！输入 'exit' 退出程序。\n")

while True:
    # 接收用户输入
    user_input = input("You: ").strip()

    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the chat. Goodbye!")
        break

    # 构建 messages 格式
    messages = [
        {"role": "user", "content": user_input}
    ]

    # 使用 chat template 构造输入
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
    except Exception as e:
        print("Error applying chat template:", e)
        continue

    # 编码输入文本并移动到设备 
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 进行文本生成
    try:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048,  # 控制最大输出长度
            do_sample=True,  # 开启采样以获得更自然的回答
            temperature=0.7,  # 控制生成多样性
            top_p=0.9  # nucleus sampling
        )
    except Exception as e:
        print("Error during generation:", e)
        continue
    # print("generated_ids:     ", tokenizer.decode(generated_ids, skip_special_tokens=True).strip("\n"))
    # 解码输出
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    print("output_ids:    ", tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n"))
    # 尝试解析 thinking 内容（可选）
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    # 输出结果
    print("thinking_content:", thinking_content)
    print("Assistant:", content)
