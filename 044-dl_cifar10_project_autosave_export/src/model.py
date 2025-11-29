"""模型结构定义模块。

这里的模型是一个非常简单的 CNN，用来演示：
- 卷积特征提取
- ReLU 非线性
- 池化降采样
- 全连接分类头

在真实项目中，你可以在这里替换成 ResNet / ViT 等更复杂的模型。
"""

import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """一个用于 CIFAR-10 的极简 CNN 网络。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 卷积特征提取模块：
        # 输入：3x32x32
        # 经过 Conv+ReLU+Conv+ReLU+MaxPool2d(2) 后，输出：64x16x16
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 输出 32x32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 输出 64x32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 下采样一半：64x16x16
        )

        # 全连接分类层：
        # 将 64x16x16 展平成一个向量，然后映射到 num_classes
        self.fc = nn.Linear(64 * 16 * 16, num_classes)

    def forward(self, x):
        # x: [B, 3, 32, 32]
        x = self.conv(x)               # [B, 64, 16, 16]
        x = x.view(x.size(0), -1)      # [B, 64*16*16]
        x = self.fc(x)                 # [B, num_classes]
        return x
