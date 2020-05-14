import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
    return dta_x, dta_y


def data_preprocessing(X, Y):
    X = X.reshape((-1, 256, 256)).astype(np.float32)
    Y = Y.reshape((-1, 1)).astype(np.float32)
    train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size=0.1, random_state=42)
    print((train_X.shape, train_Y.shape), (val_X.shape, val_Y.shape))

    train_X = train_X.reshape(-1, 1, 256, 256)
    train_X = torch.from_numpy(train_X)
    train_Y = torch.from_numpy(train_Y)

    val_X = val_X.reshape(-1, 1, 256, 256)
    val_X = torch.from_numpy(val_X)
    val_Y = torch.from_numpy(val_Y)

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

    train_losses.append(loss_train)
    val_losses.append(loss_val)

    loss_train.backward()
    optimizer.step()
    gc.collect()

    print('Epoch: ', epochs + 1, '\t', 'train loss: ', loss_train, '\t', 'val loss: ', loss_val)


if __name__ == '__main__':

    ticker_list = os.listdir('D:/GitHub/Backtesting/images_npy')
    model = res_conv1(1, 64)
    lr = 0.0001
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_losses = []
    val_losses = []

    for comp in ticker_list:
        ticker_dta = os.listdir('D:/GitHub/Backtesting/images_npy/{}'.format(comp))
        for dta in ticker_dta:
            try:
                path = 'D:/GitHub/Backtesting/images_npy/{}/{}'.format(comp, dta)
                dta_x, dta_y = dataLoader(path)
                print("Train on {}".format(dta))
                # Begin training

                train_X, train_Y, val_X, val_Y = data_preprocessing(dta_x, dta_y)
                gc.collect()

                train(0)

            except RuntimeError:
                model_path = './cnn_res.pth'
                torch.save(model.state_dict(), model_path)
                print("{} breaks the computer!!!".format(dta))
                break