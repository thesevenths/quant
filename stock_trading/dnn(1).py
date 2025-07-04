import torch
import torch.nn as nn
import torch.nn.functional as F


# 模型
class SimpleTradingDNN(nn.Module):
    def __init__(self):
        super(SimpleTradingDNN, self).__init__()
        self.fc1 = nn.Linear(302, 600)  # 输入层到隐藏层
        self.fc2 = nn.Linear(600, 3)  # 隐藏层到输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层，使用ReLU激活函数
        x = self.fc2(x)  # 输出层
        return x

