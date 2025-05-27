import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']

# 结合了 BCE（像素级准确性）+ Dice（区域重叠）优势，适合不平衡分割任务
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        """
        :param input: 模型输出 logits
        :param target: ground truth 掩码（0/1）
        :return:
        """
        # 1.BCE Loss
        # 直接使用带logits的BCE，内部自动做了sigmoid
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5

        # 2.Dice Loss
        input = torch.sigmoid(input) # 激活sigmoid，转换为概率
        # reshape 展平
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        # 计算Dice系数，Dice系数表示预测和目标之间的重叠程度，越大越好
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        return 0.5 * bce + dice

# 基于Lovasz的结构感知损失
class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        # 计算每张图的结构性错误
        loss = lovasz_hinge(input, target, per_image=True)
        return loss
