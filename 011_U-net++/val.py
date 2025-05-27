import argparse
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter

"""
需要指定参数：--name dsb2018_96_NestedUNet_woDS
"""

# 参数解析函数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='model name')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # 加载配置文件
    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 打印配置信息
    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    # 启用 cuDNN 自动优化
    cudnn.benchmark = True

    # 创建模型
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    # 加载到 CUDA（如可用）
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        device = torch.device("cpu")
        model.to(device)

    # 读取所有图像ID（去掉扩展名）
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # 拆分验证集
    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    # 加载模型权重
    model.load_state_dict(torch.load('models/%s/model.pth' % config['name']))
    model.eval() # 设置为评估模式

    # 定义验证集图像增强（大小缩放 + 归一化）
    val_transform = Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    # 加载验证集数据集
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    # 构建数据加载器
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # 初始化 IoU 评估器
    avg_meter = AverageMeter()

    # 创建输出目录
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)

    # 不计算梯度（验证阶段）
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            else:
                device = torch.device("cpu")
                input = input.to(device)
                target = target.to(device)

            # 模型前向传播
            if config['deep_supervision']:
                output = model(input)[-1] # 使用最后一个输出
            else:
                output = model(input)

            # 计算 IoU 得分并更新平均器
            iou = iou_score(output, target)
            avg_meter.update(iou, input.size(0))

            # 输出后处理（Sigmoid + 转为 numpy）
            output = torch.sigmoid(output).cpu().numpy()

            # 将预测结果保存为图像
            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))
    # 打印平均 IoU
    print('IoU: %.4f' % avg_meter.avg)

    # 可视化样例
    plot_examples(input, target, model, num_examples=3)

    # 释放显存
    torch.cuda.empty_cache()

# 可视化函数（随机展示若干输入/预测/目标图像）
def plot_examples(datax, datay, model, num_examples=6):
    fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18, 4 * num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        image_arr = model(datax[image_indx:image_indx + 1]).squeeze(0).detach().cpu().numpy()

        # 原始图像
        ax[row_num][0].imshow(np.transpose(datax[image_indx].cpu().numpy(), (1, 2, 0))[:, :, 0])
        ax[row_num][0].set_title("Orignal Image")

        # 模型预测图像
        ax[row_num][1].imshow(np.squeeze((image_arr > 0.40)[0, :, :].astype(int)))
        ax[row_num][1].set_title("Segmented Image localization")

        # Ground Truth 掩码图像
        ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1, 2, 0))[:, :, 0])
        ax[row_num][2].set_title("Target image")
    plt.show()


if __name__ == '__main__':
    main()
