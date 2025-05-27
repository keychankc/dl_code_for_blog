import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

import archs
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool
import albumentations as A
from albumentations import Compose, OneOf


ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

"""
指定参数： --dataset dsb2018_96  --arch NestedUNet
--dataset dsb2018_96 数据集
--arch NestedUNet 网络结构

"""

# 1.参数解析函数
def parse_args():
    parser = argparse.ArgumentParser()

    # 模型名称
    parser.add_argument('--name', default=None, help='model name: (default: arch+timestamp)')
    # 训练轮数
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    # 每个batch的数据量
    parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N', help='mini-batch size (default: 16)')

    # model，控制模型结构、是否使用深度监督、输入图像尺寸
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' + ' | '.join(ARCH_NAMES) + ' (default: NestedUNet)')
    # 是否使用深度监督（U-Net++ 特性），false，只用最后一层输出
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    # 输入图像通道数 3表示彩色图像（RGB）
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    # 类别数量1 表示二分类（前景 vs 背景）
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    # 输入图像宽
    parser.add_argument('--input_w', default=96, type=int,
                        help='image width')
    # 输入图像高
    parser.add_argument('--input_h', default=96, type=int,
                        help='image height')

    # 损失函数
    parser.add_argument('--loss', default='BCEDiceLoss', choices=LOSS_NAMES,
                        help='loss: ' + ' | '.join(LOSS_NAMES) + ' (default: BCEDiceLoss)')

    # 数据集
    parser.add_argument('--dataset', default='dsb2018_96', help='dataset name')
    # 图像文件后缀
    parser.add_argument('--img_ext', default='.png', help='image file extension')
    # 掩码的文件后缀
    parser.add_argument('--mask_ext', default='.png', help='mask file extension')

    # 优化器
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' + ' | '.join(['Adam', 'SGD']) + ' (default: Adam)')
    # 初始学习率
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
    # 动量项（SGD 特有）
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    # 权重衰减（正则化）
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    # 是否使用 Nesterov 动量
    parser.add_argument('--nesterov', default=False, type=str2bool, help='nesterov')

    # 学习率调度器
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    # 最小学习率
    parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
    # 每次降低的倍数，用于某些调度器
    parser.add_argument('--factor', default=0.1, type=float)
    # scheduler 的耐心（多少次不提升再调整学习率）
    parser.add_argument('--patience', default=2, type=int)
    # 在哪些 epoch 降低学习率
    parser.add_argument('--milestones', default='1,2', type=str)
    # 学习率衰减系数
    parser.add_argument('--gamma', default=2 / 3, type=float)
    # 早停 -1 表示不启用
    parser.add_argument('--early_stopping', default=-1, type=int, metavar='N', help='early stopping (default: -1)')
    # 数据加载线程数
    parser.add_argument('--num_workers', default=0, type=int)

    config = parser.parse_args()

    return config

# 训练函数 train
def train(config, train_loader, model, criterion, optimizer):

    # 记录每轮的平均 loss 和 IOU
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        else:
            device = torch.device("cpu")
            input = input.to(device)
            target = target.to(device)

        # compute output
        if config['deep_supervision']:
            outputs = model(input) # 多输出用于深度监督
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新记录器
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg),])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg)])

# 验证函数 validate
def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    # 设置为评估模式
    model.eval()

    with torch.no_grad(): # 不计算梯度，节省显存
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            else:
                device = torch.device("cpu")
                input = input.to(device)
                target = target.to(device)

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg)])

def main():
    # 获取参数字典
    config = vars(parse_args())

    # 自动生成模型名
    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    # 保存配置文件
    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # 初始化损失函数
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()  # WithLogits 就是先将输出结果经过sigmoid再交叉熵
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # 创建模型
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'])
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        device = torch.device("cpu")
        model.to(device)

    # 选择优化器
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    # 配置学习率调度器
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=True, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError


    # 加载图像ID，并划分训练/验证集
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    # 数据增强策略
    # 如何用三方库做数据增强：
    # https://github.com/albumentations-team/albumentations_examples/tree/main
    # example_kaggle_salt.ipynb
    train_transform = Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        OneOf([
            A.HueSaturationValue(p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
        ], p=1),
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])
    val_transform = Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    # 加载数据集
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)  # 不能整除的batch是否就不要了
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # 日志记录结构
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
    ])

    best_iou = 0
    trigger = 0
    # 开始训练循环
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)
        # 更新学习率
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))
        # 日志记录
        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' % config['name'], index=False)

        trigger += 1
        # 保存最佳模型
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # 早停判断
        if 0 <= config['early_stopping'] <= trigger:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()