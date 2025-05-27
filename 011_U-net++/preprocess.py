import os
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm


def main():
    # 设置目标图像大小为 96×96
    img_size = 96
    # 读取所有子文件夹，每个子文件夹对应一个图像样本（包含 images 和 masks 子目录）
    paths = glob('inputs/stage1_train/*')

    # 用于存放缩放后的图像
    os.makedirs('inputs/dsb2018_%d/images' % img_size, exist_ok=True)
    # 用于存放合并后的掩码图像
    os.makedirs('inputs/dsb2018_%d/masks/0' % img_size, exist_ok=True)

    for i in tqdm(range(len(paths))): # 用tqdm显示处理进度
        path = paths[i]
        # 读取图片
        img = cv2.imread(os.path.join(path, 'images', os.path.basename(path) + '.png'))

        # 合并掩码
        mask = np.zeros((img.shape[0], img.shape[1]))
        for mask_path in glob(os.path.join(path, 'masks', '*')):
            mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
            mask[mask_] = 1

        # 图像通道处理
        if len(img.shape) == 2: # 灰度图（2D），复制为3通道图像
            img = np.tile(img[..., None], (1, 1, 3))
        if img.shape[2] == 4: # RGBA（4通道），去掉透明度通道，只保留RGB
            img = img[..., :3]

        # 将图像和掩码都缩放为 96×96
        img = cv2.resize(img, (img_size, img_size))
        mask = cv2.resize(mask, (img_size, img_size))

        # 保存缩放后的图像和掩码
        cv2.imwrite(os.path.join('inputs/dsb2018_%d/images' % img_size, os.path.basename(path) + '.png'), img)
        cv2.imwrite(os.path.join('inputs/dsb2018_%d/masks/0' % img_size, os.path.basename(path) + '.png'), (mask * 255).astype('uint8'))

if __name__ == '__main__':
    main()