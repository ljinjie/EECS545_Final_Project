import checkpoint
from SegNet import SegNet
from dataset import KITTI
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=12, help='batch-size fo training, default=12')
parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs, default=1000')
parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum, default=0.9')
parser.add_argument('--use-cpu', action='store_true', default=False, help='use CPU only for training, default=False')
parser.add_argument('--in-channel', type=int, default=3, help='input channels, default=3')
parser.add_argument('--out-channel', type=int, default=32, help='output channels, default=32')

params = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() and (not params.use_cpu) else "cpu")

if __name__ == '__main__':
    classes = np.load()
    x_train = KITTI(classes)
    x_train_loader = torch.utils.data.DataLoader(x_train, batch_size=params.batch_size, shuffle=True, num_workers=4)

    # training
    model = SegNet()
    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=params.momentum)
    criterion = nn.CrossEntropyLoss()

    model, start_epoch = checkpoint.restore_checkpoint(model, 'checkpoints', cuda=True, force=False)


