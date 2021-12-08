import checkpoint
from SegNet import SegNet
from dataset import KITTI
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=12, help='batch-size fo training, default=4')
parser.add_argument('--epochs', type=int, default=400, help='number of training epochs, default=500')
parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum, default=0.9')
parser.add_argument('--use-cpu', action='store_true', default=False, help='use CPU only for training, default=False')
parser.add_argument('--in-channel', type=int, default=3, help='input channels, default=3')
parser.add_argument('--out-channel', type=int, default=31, help='output channels, default=31')

params = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() and (not params.use_cpu) else "cpu")


def train_epoch(train_loader, model, criterion, optimizer):
    """
        Train the model for one iteration through the training set.
    """
    loss_total = 0.0
    # idx = 1
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
    # ignore void label, class #30
    classes_ignore = np.delete(classes, 30, axis=0)
    x_train = KITTI(classes_ignore)
    x_test = KITTI(classes_ignore, data_path='data/test', label_path='label/test')
    x_train_loader = torch.utils.data.DataLoader(x_train, batch_size=params.batch_size, shuffle=True, num_workers=2)
    x_test_loader = torch.utils.data.DataLoader(x_test, batch_size=params.batch_size, shuffle=True, num_workers=2)

    # training
    model = SegNet(params.in_channel, params.out_channel).to(device)
    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=params.momentum)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

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

    plt.semilogy(range(len(stats)), stats)
    plt.title('Training Loss')
    plt.show()

    # testing
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(x_test_loader, 1):
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            prediction = torch.argmax(y_pred, dim=1).detach().cpu().numpy()
            ground_truth = torch.argmax(y, dim=1).detach().cpu().numpy()

            for j in range(prediction.shape[0]):
                img_pred = np.zeros((prediction.shape[1], prediction.shape[2], 3))
                img_true = np.zeros((prediction.shape[1], prediction.shape[2], 3))
                for m in range(classes_ignore.shape[0]):
                    mask_pred = (prediction[j] == m)
                    mask_true = (ground_truth[j] == m)
                    img_pred[mask_pred] = classes_ignore[m]
                    img_true[mask_true] = classes_ignore[m]
                img_pred /= 255.0
                img_true /= 255.0
                plt.imshow(img_pred)
                plt.show()
                plt.imshow(img_true)
                plt.show()

            if i >= 3:
                break
