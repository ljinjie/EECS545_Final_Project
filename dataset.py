import os
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms


class KITTI:
    def __init__(self, classes, batch_size=15, data_path='data', label_path='label'):
        self.classes = classes
        self.batch_size = batch_size
        self.data_path = data_path
        self.label_path = label_path
        self.transform = self.get_transforms()
        self.img_list = os.listdir(self.data_path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        data_name = self.img_list[item]
        data = io.imread(os.path.join(self.data_path, data_name))
        # currently set up for CamVid, may need changes to adapt to KITTI
        temp_idx = data_name.find('.png')
        label = self.get_label(io.imread(os.path.join(self.label_path, (data_name[:temp_idx] + '_L' + data_name[temp_idx:]))))
        data = self.transform(data)
        label = self.transform(label)
        result = (data, label)
        return result

    def get_transforms(self):
        # Normalize?
        # transform_list = [transforms.CenterCrop((720, 720)),
        #                   transforms.Resize((224, 224)),
        #                   transforms.ToTensor()]
        transform_list = [transforms.ToTensor()]
        transform = transforms.Compose(transform_list)
        return transform

    def get_label(self, img):
        # Convert an m*n*3 labeled image to an m*n*k label in one-hot encoding
        # m*n: img resolution, k: # of classes
        result = np.zeros((img.shape[0], img.shape[1], self.classes.shape[0]))
        for c in range(self.classes.shape[0]):
            current_label = np.nanmin(img == self.classes[c], axis=-1)
            result[:, :, c] = current_label
        return result
