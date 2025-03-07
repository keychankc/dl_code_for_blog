from pathlib import Path
from torchvision import transforms, models, datasets
import os
import torch
import json
from torch import nn
from torch import optim
import time
import copy
import matplotlib.pyplot as plt
import numpy as np


def load_label_name():
    with open('./data/cat_to_name.json', 'r') as f:
        return json.load(f)

def get_device():
    if not torch.cuda.is_available():
        print('CUDA is not available. Training on CPU ...')
        return 'cpu'
    else:
        print('CUDA is available! Training on GPU ...')
        return 'cuda:0'


def data_transforms():
    return {
        'train':
            transforms.Compose([
                transforms.Resize([96, 96]),  # 转化成96*96大小的图像数据
                transforms.RandomRotation(45),  # 数据增强，-45到45度之间随机旋转
                transforms.CenterCrop(64),  # 数据增强，从中心开始裁剪为64*64
                transforms.RandomHorizontalFlip(p=0.5),  # 数据增强，选择一个概率概率随机水平翻转
                transforms.RandomVerticalFlip(p=0.5),  # 数据增强，选择一个概率概率随机水平翻转
                # 数据增强，亮度，对比度，饱和度，色相调整
                transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomGrayscale(p=0.025),  # 数据增强，灰度调整
                transforms.ToTensor(),  # 转化为tensor结构
                # ImageNet提供的均值，标准差
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        'valid':
            transforms.Compose([
                transforms.Resize([64, 64]),
                transforms.ToTensor(),
                # ImageNet提供的均值，标准差
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
    }


def load_data_loader():
    batch_size = 128
    image_datasets = {x: datasets.ImageFolder(os.path.join(Path("./data/"), x), data_transforms()[x]) for x in
                      ['train', 'valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
                   ['train', 'valid']}
    return dataloaders['train'], dataloaders['valid']


# 反标准化（恢复原始图像）
def denormalize(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image.numpy().transpose((1, 2, 0))  # 转换为 HWC 格式
    image = std * image + mean  # 反标准化
    image = np.clip(image, 0, 1)  # 限制在 [0,1] 之间
    return image


def init_model(num_classes):
    # torch自带的18层训练好的模型
    model_ft = models.resnet18(weights="DEFAULT")
    for param in model_ft.parameters():
        param.requires_grad = False  # 冻结参数，使其在训练过程中 不会被优化
    model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes)  # 替换ResNet18的最后全连接层
    model_ft.to(get_device())  # 训练GPU or CPU
    return model_ft


def get_optimizer(_model_resnet):
    params_to_update = []
    for name, param in _model_resnet.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    return optim.Adam(params_to_update, lr=1e-2)


def train_one_epoch(_model, dataloader, optimizer, _criterion, device, _scheduler):
    """
    训练一个 epoch
    """
    _model.train()  # 设置为训练模式
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # 清空梯度
        outputs = _model(inputs)  # 前向传播
        _, predicted = torch.max(outputs, 1)  # 预测值
        loss = _criterion(outputs, labels)  # 计算损失

        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        _scheduler.step()  # 学习率衰减

        running_loss += loss.item() * inputs.size(0)  # 累加 batch loss
        running_corrects += torch.sum((predicted == labels).int())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def validate_one_epoch(_model, dataloader, _criterion, device):
    """
    进行一个 epoch 的验证
    """
    _model.eval()  # 设置为评估模式
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():  # 不计算梯度，加速计算
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = _model(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = _criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum((predicted == labels).int())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def train_model(_model, train_loader, valid_loader, _criterion, optimizer, _scheduler, num_epochs, device, _model_name):
    """
    训练完整模型
    """
    start_time = time.time()
    best_acc = 0.0
    best_model_wts = copy.deepcopy(_model.state_dict())

    train_acc_history = []
    train_losses = []
    valid_acc_history = []
    valid_losses = []

    learning_rates = [optimizer.param_groups[0]['lr']]

    for epoch in range(num_epochs):
        print(f'Epoch:{epoch + 1}/{num_epochs}')

        # 训练
        train_loss, train_acc = train_one_epoch(_model, train_loader, optimizer, _criterion, device, _scheduler)
        train_losses.append(train_loss)
        train_acc_history.append(train_acc)

        # 验证
        valid_loss, valid_acc = validate_one_epoch(_model, valid_loader, _criterion, device)
        valid_losses.append(valid_loss)
        valid_acc_history.append(valid_acc)

        time_elapsed = time.time() - start_time
        print(f"Time elapsed: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Valid Loss: {valid_loss:.4f} Acc: {valid_acc:.4f}")

        # 记录最佳模型
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model_wts = copy.deepcopy(_model.state_dict())
            state = {
                'state_dict': _model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, model_name)

        # 记录学习率
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        learning_rates.append(optimizer.param_groups[0]['lr'])
        _scheduler.step()  # 学习率衰减
        print('-' * 100)

    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation Acc: {best_acc:.4f}')

    # 用最佳模型的参数作为最终模型
    _model.load_state_dict(best_model_wts)
    return _model, train_losses, train_acc_history, valid_losses, valid_acc_history, learning_rates

def second_train(_model_name, _train_dl, _valid_dl, _criterion):
    _model = init_model(num_classes=102)
    checkpoint = torch.load(_model_name, weights_only=True)
    _model.load_state_dict(checkpoint['state_dict'])
    for param in _model.parameters():
        param.requires_grad = True
    _optimizer_next = optim.Adam(_model.parameters(), lr=1e-3)
    _scheduler_next = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    train_model(_model, _train_dl, _valid_dl, _criterion, _optimizer_next, _scheduler_next, 10,
                get_device(), _model_name)

def show_predicted(_model_name):
    _model = init_model(num_classes=102)
    checkpoint = torch.load(_model_name, weights_only=True)
    _model.load_state_dict(checkpoint['state_dict'])
    label_name = load_label_name()
    images, labels = next(iter(valid_dl))
    _model.eval()
    output = _model(images)
    _, predicted_tensor = torch.max(output, 1)
    train_on_gpu = torch.cuda.is_available()
    predicted = np.squeeze(predicted_tensor.numpy()) if not train_on_gpu else np.squeeze(predicted_tensor.cpu().numpy())
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))  # 创建 2x4 的子图网格
    axes = axes.flatten()
    for i in range(8):
        img = denormalize(images[i])  # 反标准化
        axes[i].imshow(img)  # 显示图片
        axes[i].axis("off")  # 关闭坐标轴
        predicted_label = label_name.get(str(predicted[i]), "Unknown")
        true_label = label_name.get(str(labels[i].item()), "Unknown")
        title_color = "green" if predicted_label == true_label else "red"
        axes[i].set_title("{} ({})".format(predicted_label, true_label), color=(
            title_color))  # 显示标签
    plt.tight_layout()
    plt.show()

def draw_dl_8img(data_loader):
    images, labels = next(iter(data_loader))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))  # 创建 2x4 的子图网格
    axes = axes.flatten()
    for i in range(8):
        img = denormalize(images[i])  # 反标准化
        axes[i].imshow(img)  # 显示图片
        axes[i].axis("off")  # 关闭坐标轴
        axes[i].set_title(f"Label: {labels[i].item()}")  # 显示标签
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 训练 测试
    train_dl, valid_dl = load_data_loader()
    # draw_dl_8img(train_dl)
    # label -> name
    label_name_json = load_label_name()
    # 模型
    model = init_model(num_classes=102)
    # 模型名称
    model_name = "best.pt"
    # 优化器
    optimizer_ft = get_optimizer(model)
    # 学习率每10次*0.1
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    # 损失函数 交叉熵
    criterion = nn.CrossEntropyLoss()
    # 一次训练
    # train_model(model, train_dl, valid_dl, criterion, optimizer_ft, scheduler, 20, get_device(), model_name)
    # 二次训练
    # second_train(model_name, train_dl, valid_dl, criterion)
    # 预测
    show_predicted(model_name)

