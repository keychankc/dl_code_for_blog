import os
import cv2
import numpy as np
import torch.utils.data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        :param img_ids:图像文件名的ID列表
        :param img_dir:图像目录路径
        :param mask_dir:掩码目录路径
        :param img_ext:图像文件扩展名（如 .png、.jpg）
        :param mask_ext:掩码扩展名（如 .png）
        :param num_classes:类别数，用于加载多个通道的掩码
        :param transform:数据增强方法
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    # 返回样本数
    def __len__(self):
        return len(self.img_ids)

    # 获取单个样本
    def __getitem__(self, idx):
        # 获取当前图像 ID
        img_id = self.img_ids[idx]
        # 读取图像
        img = cv2.imread(str(os.path.join(self.img_dir, img_id + self.img_ext)))

        # 读取对应掩码（支持多类）
        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(str(os.path.join(self.mask_dir, str(i),
            img_id + self.mask_ext)), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        # 数据增强
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # 数据归一化 & 转换格式
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        return img, mask, {'img_id': img_id}
