import torch.nn as nn

class Mnist_CNN(nn.Module):
    def __init__(self):
        super(Mnist_CNN, self).__init__()
        # (1, 28, 28) -> (16, 28, 28) ->  (16, 14, 14) -> (32, 14, 14) -> (32, 7, 7) -> (64, 7, 7)
        # in_channels 灰度图
        # out_channels 输出特征图
        # kernel_size 卷积核大小
        # stride 步长
        # padding 如果希望卷积后大小跟原来一样，需要设置padding=(kernel_size-1)/2 else stride=1
        self.conv1 = nn.Sequential(         # 输入大小 (1, 28, 28)
            # 输出的特征图为 (16, 28, 28)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),                      # relu层
            nn.MaxPool2d(kernel_size=2),    # 进行池化操作（2x2 区域）, (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # 下一个套餐的输入 (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # 输出 (32, 14, 14)
            nn.ReLU(),                      # relu层
            nn.MaxPool2d(2),                # 输出 (32, 7, 7)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),                      # 输出 (64, 7, 7)
        )
        self.out = nn.Linear(64 * 7 * 7, 10)   # 全连接层得到的结果

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)           # flatten操作：(batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output
