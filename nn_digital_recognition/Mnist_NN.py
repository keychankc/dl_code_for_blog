from torch import nn
import torch.nn.functional as F

class Mnist_NN(nn.Module):

    def __init__(self):
        super().__init__()
        # 定义属性
        self.hidden1 = nn.Linear(784, 128)   # 第一层全连接层 输入784个特征 -> 输出128个神经元
        self.hidden2 = nn.Linear(128, 256)   # 第二层全连接层 128 -> 256
        self.out = nn.Linear(256, 10)        # 输出层 256 -> 10分类
        self.dropout = nn.Dropout(0.5)                             # Dropout 层，用于防止过拟合，在每次前向传播中，随机丢弃 50% 的神经元

    # 前向传播
    def forward(self, x):
        x = F.relu(self.hidden1(x))                                # 第一层全连接层的输出经过 ReLU 激活函数。ReLU 将所有负值转换为 0，正值保持不变，引入非线性
        x = self.dropout(x)                                        # 将中间层的输出送入 Dropout 层，随机丢弃一部分神经元，以减小过拟合的风险
        x = F.relu(self.hidden2(x))                                # 第二层全连接层的输出经过 ReLU 激活函数
        x = self.dropout(x)                                        # 再次进行 Dropout 操作，丢弃一些神经元
        x = self.out(x)                                            # 输出层给出10个类别的预测值（数字0到9），每个值代表该数字的得分或概率
        return x
