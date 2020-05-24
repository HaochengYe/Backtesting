import os

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.autograd import Variable

import gc

from ResNet_CNN import *

import torch

from torch.optim import Adam
from torch import nn


def dataLoader(path):
    loaded = np.load(path)
    dta_x = loaded['x']
    dta_y = loaded['y']
    y_T = dta_y.T
    y_int = y_T.dot(1 << np.arange(y_T.shape[-1]-1, -1, -1))
    return dta_x, y_int


def data_preprocessing(X, Y):
    d = X.shape[0]
    X = X.reshape((-1, d, d)).astype(np.float32)
    train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size=0.1)
    # print((train_X.shape, train_Y.shape), (val_X.shape, val_Y.shape))

    train_X = train_X.reshape(-1, 1, d, d)
    train_X = torch.from_numpy(train_X)
    train_Y = torch.from_numpy(train_Y)
    train_Y = train_Y.type(torch.long)

    val_X = val_X.reshape(-1, 1, d, d)
    val_X = torch.from_numpy(val_X)
    val_Y = torch.from_numpy(val_Y)
    val_Y = val_Y.type(torch.long)

    return train_X, train_Y, val_X, val_Y

def train(epochs):
    # dataset
    x_train, y_train = Variable(train_X), Variable(train_Y)
    x_val, y_val = Variable(val_X), Variable(val_Y)

    optimizer.zero_grad()

    output_train = model(x_train)
    output_val = model(x_val)

    loss_train = criterion(output_train, y_train.type(torch.float))
    loss_val = criterion(output_val, y_val.type(torch.float))

    loss_train.backward()
    optimizer.step()
    gc.collect()

    print('Epoch: ', epochs + 1, '\t', 'train loss: ', round(loss_train.item(), 6), '\t', 'val loss: ', round(loss_val.item(), 6))


def visualize_train_val(train_losses, val_losses):
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(['Train Loss', 'Val Loss'])
    plt.show()


if __name__ == '__main__':
    ticker_list = os.listdir('D:/GitHub/Backtesting/images_npy')
    model = res_conv(1, 16, deepths=[1,1,1], blocks_sizes=[32,64,128])
    lr = 0.001
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if os.path.exists('ResNet_CNN.pth'):
        model.load_state_dict(torch.load('./ResNet_CNN.pth'))
        print("Reload model completed!")

    try:
        for comp in ticker_list:
            ticker_dta = os.listdir('D:/GitHub/Backtesting/images_npy/{}'.format(comp))
            for dta in ticker_dta:
                path = 'D:/GitHub/Backtesting/images_npy/{}/{}'.format(comp, dta)
                dta_x, dta_y = dataLoader(path)
                print("Train on {}!".format(dta))
                print(dta_x.shape, dta_y.shape)
                # Begin training

                for epoch in range(5):
                    train_X, train_Y, val_X, val_Y = data_preprocessing(dta_x, dta_y)
                    gc.collect()
                    train(epoch)

                # visualize_train_val(train_losses, val_losses)

                model_path = './ResNet_CNN.pth'
                torch.save(model.state_dict(), model_path)

                print("Finished training on {}!".format(dta))

    except RuntimeError:
        model_path = './ResNet_CNN.pth'
        torch.save(model.state_dict(), model_path)
        print("Breaks the computer!!!")
