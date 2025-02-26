import torch

from pathlib import Path
import gzip
import pickle

from matplotlib import pyplot

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from torch import optim
import numpy as np
import torch.nn.functional as F

from nn_digital_recognition.Mnist_NN import Mnist_NN


def get_torch_version():
    return torch.__version__

def load_mnist():
    path = Path("../data/") / "01_mnist"
    filename = "mnist.pkl.gz"
    with gzip.open((path / filename).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    return x_train, y_train, x_valid, y_valid


def print_mnist_info():
    print(train_data.shape, train_result.shape, valid_data.shape, valid_result.shape)
    print(train_data[0])
    print(train_result[:10])


def draw_mnist_data(train_d):
    fig, axes = pyplot.subplots(3, 8, figsize=(12, 6))  # 3 行 8 列，调整 fig size 以便看得更清楚
    for i in range(3):
        for j in range(8):
            ax = axes[i, j]
            ax.imshow(train_d[i * 8 + j].reshape((28, 28)), cmap="gray")
            ax.axis('off')
    pyplot.tight_layout()
    pyplot.show()

def get_tensor_dataset(batch):
    # TensorDataset：将输入数据和标签组合成一个数据集，可以方便地用来处理训练集和验证集
    # DataLoader：用来批量加载数据，支持多种功能，比如按批次加载数据、随机打乱数据、并行加载等
    train_ds = TensorDataset(train_data, train_result)
    valid_ds = TensorDataset(valid_data, valid_result)
    return DataLoader(train_ds, batch_size=batch, shuffle=True), DataLoader(valid_ds, batch_size=bs * 2)


def loss_batch(m, loss_func, xb, yb, opt=None):
    loss = loss_func(m(xb), yb)

    if opt is not None:
        loss.backward()     # 反向传播
        opt.step()          # 更新模型参数（权重和偏置）
        opt.zero_grad()     # 梯度清空

    return loss.item(), len(xb)

def fit(steps, m, loss_func, opt, train_dl, valid_dl):
    for step in range(steps):
        m.train()  # 设置为训练模式
        for xb, yb in train_dl:
            loss_batch(m, loss_func, xb, yb, opt)

        m.eval()  # 设置为验证模式
        with torch.no_grad():  # 禁用梯度计算
            losses, nums = zip(*[loss_batch(m, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print('当前step:' + str(step), '验证集损失：' + str(val_loss))


if __name__ == '__main__':
    bs = 64
    # 读取数据
    train_data, train_result, valid_data, valid_result = load_mnist()
    # 将数据转化为tensor
    train_data, train_result, valid_data, valid_result = map(torch.tensor, (train_data, train_result, valid_data, valid_result))
    # 数据打包 数据处理
    train_dataloader, valid_dataloader = get_tensor_dataset(bs)
    # 定义模型
    model = Mnist_NN()
    # 损失函数
    loss_func = F.cross_entropy
    # 优化器
    opt1 = optim.SGD(model.parameters(), lr=0.001)  # 优化器
    opt2 = optim.Adam(model.parameters(), lr=0.001)
    # 训练
    fit(25, model, loss_func, opt2, train_dataloader, valid_dataloader)
    # 验证数据
    correct = 0
    total = 0
    for xb, yb in valid_dataloader:
        outputs = model(xb)
        _, pred = torch.max(outputs.data, 1)
        total += yb.size(0)
        correct += (pred == yb).sum().item()
    print('测试集10000张图片正确率：%d %%' % (100 * correct / total))

