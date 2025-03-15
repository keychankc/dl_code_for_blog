from torchvision import datasets, transforms
import torch
import torch.nn as nn

import torch.optim as optim

from cnn_digital_recognition.Mnist_CNN import Mnist_CNN


def load_dataset():
    train_ds = datasets.MNIST(root='./data',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

    test_ds = datasets.MNIST(root='./data',
                             train=False,
                             transform=transforms.ToTensor())
    return train_ds, test_ds


def load_loader(train_ds, test_ds):
    train_dl = torch.utils.data.DataLoader(dataset=train_ds,
                                           batch_size=batch_size,
                                           shuffle=True)
    test_dl = torch.utils.data.DataLoader(dataset=test_ds,
                                          batch_size=batch_size,
                                          shuffle=True)
    return train_dl, test_dl


def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


def train(model, train_loader, test_loader, num_epochs):
    for epoch in range(num_epochs):
        # 当前epoch的结果保存下来
        train_rights = []

        for batch_idx, (data, target) in enumerate(train_loader):  # 针对容器中的每一批进行循环
            model.train()
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            right = accuracy(output, target)
            train_rights.append(right)

            if batch_idx % 100 == 0:

                model.eval()
                val_rights = []

                for (data, target) in test_loader:
                    output = model(data)
                    right = accuracy(output, target)
                    val_rights.append(right)

                # 准确率计算
                train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
                val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

                # 训练进度
                train_percent = 100 * batch_idx / len(train_loader)
                # 训练集准确率
                train_set_accurate = 100 * train_r[0].numpy() / train_r[1]
                # 测试集正确率
                test_set_accurate = 100 * val_r[0].numpy() / val_r[1]

                print(f"""当前epoch:{epoch+1} 训练进度:{train_percent:.0f}% 损失:{loss.data:.6f} 训练集准确率: {train_set_accurate:.2f}% 测试集正确率: {test_set_accurate:.2f}% """)


if __name__ == '__main__':
    # 定义超参数
    input_size = 28  # 图像的总尺寸28*28
    num_epochs = 3  # 训练的总循环周期
    batch_size = 64  # 一个批次的大小，64张图片

    # 训练集 测试机
    train_dataset, test_dataset = load_dataset()
    train_loader, test_loader = load_loader(train_dataset, test_dataset)

    # 模型
    model = Mnist_CNN()
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 定义优化器，普通的随机梯度下降算法
    # 训练
    train(model, train_loader, test_loader, num_epochs)
