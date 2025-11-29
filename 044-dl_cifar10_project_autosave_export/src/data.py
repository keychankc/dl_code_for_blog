"""数据加载模块：负责构建 CIFAR-10 的 DataLoader。

只做一件事：根据配置返回 train_loader 和 val_loader。
模型 / 训练逻辑不应该写在这里。
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def build_dataloader(batch_size: int, num_workers: int, data_dir: str = "./data"):
    """构建带有基础数据增强的 CIFAR-10 训练 & 验证 DataLoader。

    Args:
        batch_size: 每个 batch 的样本数量。
        num_workers: DataLoader 后台进程数，用来并行加载数据。
        data_dir: 数据存放路径，如果不存在会自动下载。

    Returns:
        train_loader, val_loader
    """

    # 训练集数据增强：随机裁剪 + 随机水平翻转 + 转为张量
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # 测试/验证集一般只做最基础的 ToTensor
    transform_val = transforms.Compose([
        transforms.ToTensor(),
    ])

    # CIFAR10 训练集
    train_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train,
    )

    # CIFAR10 测试集（这里作为验证集使用）
    val_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_val,
    )

    # 训练集 DataLoader，shuffle=True 打乱数据
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,   # 配合 GPU，可以加速 CPU->GPU 的内存拷贝
    )

    # 验证集 DataLoader，注意不需要 shuffle
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
