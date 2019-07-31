from __future__ import print_function

import os
import os.path as osp
import torch
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image


from utils.vis import plot_image

# class ImgAugTransform:
#     import imgaug.augmenters as iaa
#     """
#     数据扩充，由于输入图像已经历人脸对齐操作，因此扩充方式具有局限性.
#     """
#
#     def __init__(self):
#         self.aug = iaa.Sequential([
#             iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
#             iaa.GammaContrast((0.3, 2)),
#         ])
#
#     def __call__(self, img):
#         img = np.array(img)
#         return self.aug.augment_image(img)


class MultiTaskDataset(Dataset):
    """
    DD项目多分类数据处理.
    """

    def __init__(self, image_path, csv_path, img_size, transform=None):
        self.image_path = image_path
        self.csv_path = csv_path
        self.transform = transform
        self.img_size = img_size
        self.image_path_list = sorted(os.listdir(self.image_path))
        self.csv_path_list = sorted(os.listdir(self.csv_path))

        assert len(self.image_path_list) == len(self.csv_path_list)

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        # 输入原始大小的图片
        img = cv.imread(osp.join(self.image_path, self.image_path_list[idx]))
        img = cv.resize(img, (self.img_size, self.img_size))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)

        if self.transform:
            img = self.transform(Image.fromarray(img)).float()

        data = pd.read_csv(osp.join(self.csv_path, self.csv_path_list[idx]), index_col=0)

        # 读取csv文件人脸各属性标签
        face, mouth, eyebrow, eye, nose, jaw = torch.from_numpy(np.array(data.iloc[0, 0])).long(), \
                                               torch.from_numpy(np.array(data.iloc[1, 0])).long(), \
                                               torch.from_numpy(np.array(data.iloc[2, 0])).long(), \
                                               torch.from_numpy(np.array(data.iloc[3, 0])).long(), \
                                               torch.from_numpy(np.array(data.iloc[4, 0])).long(), \
                                               torch.from_numpy(np.array(data.iloc[5, 0])).long()

        return img, face, mouth, eyebrow, eye, nose, jaw


# if __name__ == '__main__':
#     dataset = MultiTaskDataset(image_path='./data/images', csv_path='./data/labels', img_size=160, transform=None)
#     if not isinstance(dataset, MultiTaskDataset):
#         raise TypeError("不属于MultiTaskDataset类...")
#     plot_image(dataset)
