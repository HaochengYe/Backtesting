import os
import yfinance as yf
from datetime import datetime
import pandas as pd
import pytz
import logging
import numpy as np
import plotly.graph_objects as go
from PIL import Image

import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.autograd import Variable

from torch.optim import Adam, SGD
import gc

from ResNet_CNN import *

import math as m
import torch
from torch.nn import Linear, ReLU, Conv1d, Conv2d, Flatten, Sequential, CrossEntropyLoss, MSELoss, MaxPool1d, MaxPool2d, Dropout, BatchNorm1d, BatchNorm2d

from torch.optim import Adam
from torch import nn
import torchvision
from functools import partial
from collections import OrderedDict

import torchvision.models as models

from torchsummary import summary


ticker_list = os.listdir('C:/Users/Andyy/Documents/GitHub/Backtesting/images_npy')

