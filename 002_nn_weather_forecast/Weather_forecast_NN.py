import torch.nn as nn

class Weather_forecast_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Weather_forecast_NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)    # 第一层：输入层 -> 隐藏层
        self.sigmoid = nn.Sigmoid()                      # Sigmoid 激活函数
        self.fc2 = nn.Linear(hidden_size, output_size)   # 第二层：隐藏层 -> 输出层

    def forward(self, x):
        x = self.fc1(x)       # 第一层
        x = self.sigmoid(x)   # 激活函数
        x = self.fc2(x)       # 第二层
        return x
