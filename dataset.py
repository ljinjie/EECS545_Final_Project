import os
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms


class KITTI:
    def __init__(self, classes, batch_size=12, data_path='data/train', label_path='label/train'):
        self.classes = classes
        self.batch_size = batch_size
        self.data_path = data_path
        self.label_path = label_path
        self.train_dataset = self.get_train_numpy()
        self.x_mean, self.x_std = self.compute_train_statistics()
        self.transform_data = self.get_transforms_data()
        self.transform_label = self.get_transforms_label()
        self.img_list = os.listdir(self.data_path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        data_name = self.img_list[item]
        data = io.imread(os.path.join(self.data_path, data_name))
        # currently set up for CamVid, may need changes to adapt to KITTI
        temp_idx = data_name.find('.png')
        label = self.get_label(io.imread(os.path.join(self.label_path, (data_name[:temp_idx] + '_L' + data_name[temp_idx:]))))
        data = self.transform_data(data)
        label = self.transform_label(label)
        result = (data, label)
        return result

    def get_transforms_data(self):
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(self.x_mean, self.x_std)]
        transform = transforms.Compose(transform_list)
        return transform

    def get_transforms_label(self):
        transform_list = [transforms.ToTensor()]
        transform = transforms.Compose(transform_list)
        return transform

    def get_label(self, img):
        # Convert an m*n*3 labeled image to an m*n*k label in one-hot encoding
        # m*n: img resolution, k: # of classes
        result = -np.ones((img.shape[0], img.shape[1], self.classes.shape[0]))

        for c in range(self.classes.shape[0]):
            current_label = np.nanmin(img == self.classes[c], axis=-1)
            result[:, :, c] = current_label

        return result

    def get_train_numpy(self):
        filenames = [name for name in os.listdir(self.data_path)]
        train_x = np.zeros((len(filenames), 360, 480, 3))
        for i, filename in enumerate(filenames):
            train_x[i] = np.array(plt.imread(os.path.join(self.data_path, filename)))
        # print(train_x)

        return train_x

    def compute_train_statistics(self):
        x_mean = np.mean(np.array(self.train_dataset), axis=(0, 1, 2))  # per-channel mean
        x_std = np.std(np.array(self.train_dataset), axis=(0, 1, 2))  # per-channel std
        return x_mean, x_std
