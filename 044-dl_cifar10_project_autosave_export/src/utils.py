"""辅助工具模块。当前只包含随机种子设置。"""

import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """设定所有相关库的随机种子，保证实验可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 让 CuDNN 以确定性模式运行（会稍微牺牲一点性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
