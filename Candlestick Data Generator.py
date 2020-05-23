import os
import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
import pytz
import logging
import plotly.graph_objects as go
from PIL import Image

import matplotlib.pyplot as plt


def init_logging():
    global tz
    tz = pytz.timezone('US/Eastern')
    if not os.path.exists('log'):
        os.makedirs('log')

    date = datetime.now(tz).date().strftime('%Y%m%d')
    logging.basicConfig(filename='log/{}.log'.format(date), level=logging.INFO)

    return None


def is_valid(data):
    # check if the entry is valid
    if isinstance(data, float):
        return data
    else:
        return 1e-5


def momentum_ret(data):
    x_0 = is_valid(data['Close'][0])
    x_T = is_valid(data['Close'][-1])
    mid = data.shape[0] // 2
    x_t = is_valid(data['Close'][mid])
    
    ttl_ret = (x_T - x_0) / x_0
    half_ret = (x_t - x_0) / x_0
    return ttl_ret + half_ret


def mean_cutoff(data):
    mu = data['Close'].mean()
    upr = (data['Close'] > mu).sum()
    lwr = (data['Close'] <= mu).sum()
    return upr - lwr


def half_return_diff(data):
    x_0 = is_valid(data['Close'][0])
    x_T = is_valid(data['Close'][-1])
    mid = data.shape[0] // 2
    x_t = is_valid(data['Close'][mid])
    
    first_ret = (x_t - x_0) / x_0
    second_ret = (x_T - x_t) / x_t
    return second_ret - first_ret


def consec_trend(data):
    arr = np.where(data['Close'] > data['Open'], 1, 0)
    count_1 = 0
    count_0 = 0

    for i in range(len(arr)-1):
        if (arr[i] == 1) & (arr[i+1] == 1):
            count_1 += 1

        elif (arr[i] == 0) & (arr[i+1] == 0):
            count_0 += 1
    
    return count_1 - count_0


def dta_to_candlestick(data):
    l = len(data)
    # Make candlestick picture
    layout = go.Layout(xaxis=dict(ticks='',
                                  showgrid=False,
                                  showticklabels=False,
                                  rangeslider=dict(visible=False)),
                       yaxis=dict(ticks='',
                                  showgrid=False,
                                  showticklabels=False),
                       width=256,
                       height=256,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)')
    fig = go.Figure(data=[go.Candlestick(x=np.linspace(1,l,l),
                                         open=data.Open,
                                         high=data.High,
                                         low=data.Low,
                                         close=data.Close)],
                    layout=layout)
    fig.write_image("images/fig-1.png")

    # Convert to numpy array
    im = Image.open('images/fig-1.png')

    # im = im.resize((300,300),Image.ANTIALIAS)
    data = np.asarray(im)

    # Return the first channel of the image
    return data[:, :, 0]


def dta_transformation(data, est_h):
    # Make sure data has sufficient columns
    assert 'Open' in data.columns
    assert 'High' in data.columns
    assert 'Low' in data.columns
    assert 'Close' in data.columns

    x = []
    y = []

    for i in range(est_h, data.shape[0]):
        sub_dta = data.iloc[i - est_h:i]

        mom_ret = int(momentum_ret(sub_dta)>0)
        mean_cut = int(mean_cutoff(sub_dta)>0)
        half_ret = int(half_return_diff(sub_dta)>0)
        consec = int(consec_trend(sub_dta)>0)
        
        y_i = [mom_ret, mean_cut, half_ret, consec]
        x_i = dta_to_candlestick(sub_dta)

        y.append(y_i)
        x.append(x_i)

        print("{}/{}".format(i - est_h, data.shape[0] - est_h))

    return x, y


if __name__ == '__main__':

    txt_file = open('ticker_list.txt', 'r')
    sp500_list = [line.rstrip('\n') for line in txt_file]

    START = datetime(1980, 1, 1)
    END = datetime(2020, 4, 23)
    init_logging()

    for ticker in sp500_list[58:100]:
        tic = yf.Ticker(ticker)
        hist = tic.history(start=START, end=END)
        if hist.shape[0] > 1000:
            x, y = dta_transformation(hist, 10)
            x = np.stack(x, axis=2)
            y = np.stack(y, axis=1)

            L = x.shape[2] // 1000
            remainder = x.shape[2] % 1000

            for i in range(L):
                sub_x = x[:, :, (1000 * i + remainder):(1000 * (i + 1) + remainder)]
                sub_y = y[:, (1000 * i + remainder):(1000 * (i + 1) + remainder)]
                if not os.path.exists('images_npy-1/{}'.format(ticker)):
                    os.makedirs('images_npy-1/{}'.format(ticker))
                np.savez_compressed('images_npy-1/{}/{}_{}'.format(ticker, ticker, i), x=sub_x, y=sub_y)

            print("\n")
            status = "Successfully retrieve {}'s data.".format(ticker)
            print(status)
            logging.info(status)
            print("\n")

        else:
            print("\n")
            status = "Insufficient data for {}.".format(ticker)
            print(status)
            logging.info(status)
            print("\n")

