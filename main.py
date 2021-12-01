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
parser.add_argument('--batch-size', type=int, default=4, help='batch-size fo training, default=4')
parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs, default=1000')
parser.add_argument('--learning-rate', type=float, default=0.000001, help='learning rate, default=0.000001')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum, default=0.9')
parser.add_argument('--use-cpu', action='store_true', default=False, help='use CPU only for training, default=False')
parser.add_argument('--in-channel', type=int, default=3, help='input channels, default=3')
parser.add_argument('--out-channel', type=int, default=32, help='output channels, default=32')

params = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() and (not params.use_cpu) else "cpu")


def train_epoch(train_loader, model, criterion, optimizer):
    """
        Train the model for one iteration through the training set.
    """
    loss_total = 0.0
    idx = 1
    for idx, (X, y) in enumerate(train_loader, 1):
        # clear parameter gradients
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

        loss_total += loss.item()

    return loss_total / (idx * train_loader.batch_size)


if __name__ == '__main__':
    classes = np.load('classes.npy')
    x_train = KITTI(classes)
    x_train_loader = torch.utils.data.DataLoader(x_train, batch_size=params.batch_size, shuffle=True, num_workers=4)

    # training
    model = SegNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=params.momentum)
    criterion = nn.CrossEntropyLoss()

    model, start_epoch, stats = checkpoint.restore_checkpoint(model, 'checkpoints', cuda=True, force=False)

    for ep in range(start_epoch, params.epochs):
        print('Running Epoch {}:'.format(ep))
        loss = train_epoch(x_train_loader, model, criterion, optimizer)
        stats.append(loss)
        print('\tTraining loss: {}'.format(loss))
        print('\tSaving checkpoint...')
        print()
        checkpoint.save_checkpoint(model, ep + 1, 'checkpoints', stats)

    print('-------------------')
    print('Training completed.')
    print('-------------------')
