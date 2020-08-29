import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from datetime import datetime
import matplotlib.pyplot as plt

from scipy.stats import kurtosis, iqr
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import arch

# ML imports
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
import keras.backend as K
from keras.callbacks import EarlyStopping
from tensorflow.keras import initializers

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import coint

def data_preprocess(dta):
    dta['Date'] = pd.to_datetime(dta['Date'], format='%Y-%m-%d')
    dta = dta.set_index(dta['Date'])
    # NHLI not traded
    dta.drop(['Date', 'NHLI'], axis=1, inplace=True)
    dta.dropna(how='all', inplace=True)
    for tick in dta.columns:
        tick_series = dta[tick]
        start_pos = tick_series.first_valid_index()
        valid_series = tick_series.loc[start_pos:]
        if valid_series.isna().sum() > 0:
            dta.drop(tick, axis=1, inplace=True)

    for tick in dta.columns:
        dta[tick] = dta[tick].mask(dta[tick] == 0).ffill(downcast='infer')

    return dta[dta.index >= dta['SPY'].first_valid_index()]


def coin_group(tick, dta):
    """
    Use cointegration test and correlation to find predictive stocks for target
    :param tick: string for the target stock
    :param dta: the data file (csv) that contains the tick
    :return: a list of tickers that are in sp500 which predict the target
    """
    original_series = dta[tick]

    if tick in sp.columns:
        temp = pd.concat([sp.drop([tick], axis=1), original_series], axis=1).dropna(axis=1)
    else:
        temp = pd.concat([sp, original_series], axis=1).dropna(axis=1)

    temp['%s_LAG' % tick] = temp[tick].shift(-120)
    temp.dropna(inplace=True)

    y = temp['%s_LAG' % tick]
    cointegrat = {}
    correlat = {}

    for i in temp.columns[:-2]:
        x = temp[i]
        score, pval, _ = coint(x, y, trend='ct')
        corr = x.corr(y)

        cointegrat[i] = pval
        correlat[i] = corr

    best_coint = sorted(cointegrat, key=cointegrat.get)[:10]
    best_corr = sorted(correlat, key=correlat.get, reverse=True)[:10]

    intersect = list(set(best_coint) & set(best_corr))

    return intersect


sp = pd.read_csv('sp500_stock.csv')
dta = pd.read_csv('broader_stock.csv')

sp = data_preprocess(sp)
dta = data_preprocess(dta)





