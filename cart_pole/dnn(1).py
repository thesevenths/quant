import torch
import torch.nn as nn
import torch.nn.functional as F


# 模型
class CartPoleDNN(nn.Module):
    def __init__(self):
        super(CartPoleDNN, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # 输入层到隐藏层
        self.fc2 = nn.Linear(128, 2)  # 隐藏层到输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层，使用ReLU激活函数
        x = self.fc2(x)  # 输出层
        return x


# model = torch.nn.Sequential(
#     torch.nn.Linear(4, 128),
#     torch.nn.ReLU(),
#     torch.nn.Linear(128, 2),
# )
