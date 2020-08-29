import yfinance as yf

from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ML imports
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
import keras.backend as K
from keras.callbacks import EarlyStopping
from tensorflow.keras import initializers

from sklearn.model_selection import train_test_split

import statsmodels.api as sm
import pandas as pd
import numpy as np

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


def coeff_deter(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred) * 1e6)
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)) * 1e6)
    return 1 - SS_res / (SS_tot + K.epsilon())


def coint_group(tick, dta):
    """
    Use cointegration test and correlation to find predictive stocks for target
    :param tick: string for the target stock
    :param dta: the data file (csv) that contains the tick
    :return: a list of tickers that are in sp500 which predict the target
    """
    original_series = dta[tick]

    if tick in sp.columns:
        temp = pd.concat([sp.drop([tick], axis=1), original_series], axis=1)
        temp = temp[temp[tick].notnull()].dropna(axis=1)
    else:
        temp = pd.concat([sp, original_series], axis=1)
        temp = temp[temp[tick].notnull()].dropna(axis=1)

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
    print("There are {} cointegrated stocks.".format(len(intersect)))
    return intersect, temp


def train_profit(tick, fitted_val, dta):
    inventory = 0
    asset = 0
    record = [0]
    forecast_diff = fitted_val
    T = min(len(forecast_diff), len(dta))

    for t in range(T):
        trend_good = forecast_diff[t] > dta[tick].iloc[t]
        price = dta[tick].iloc[t]
        if trend_good and inventory == 0:
            # buy
            asset -= price
            inventory += 1
        elif not trend_good and inventory == 1:
            # sell
            asset += price
            inventory -= 1
        elif t == len(forecast_diff) - 1 and inventory == 1:
            asset += price
            inventory -= 1
        else:
            asset = record[-1]
        record.append(asset)

    return asset, record


def regression_mod(Y, dta):
    """
    Use basic regression model to forecast
    :param X: list of strings of tickers
    :param Y: string of lagged target ticker
    :param dta: the data set that contains X and Y
    :return: the regression model (statsmodels mod format)
    """
    X = dta[coint_corr]
    Y = dta[Y]
    mod = sm.OLS(Y, sm.add_constant(X)).fit()
    return mod


def LSTM_mod(X, Y, scaler_x, scaler_y):
    """
    To adjust lstm machine learning model architecture (layers, activations, kernels...)
    :param X: np arrays
    :param Y: np array (1 dimensional)
    :param scaler_x: a scaler class from sklearn (unfitted)
    :param scaler_y: a scaler class from sklearn (unfitted)
    :return:
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

    scaler_x = scaler_x.fit(X_train)
    scaler_y = scaler_y.fit(Y_train)

    X_train = scaler_x.transform(X_train)
    Y_train = scaler_y.transform(Y_train)

    X_test = scaler_x.transform(X_test)
    Y_test = scaler_y.transform(Y_test)

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    initializer = initializers.glorot_normal(seed=42)
    model = Sequential()
    model.add(LSTM(20, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_initializer=initializer))
    model.add(Dropout(0.4))
    model.add(Dense(10, kernel_initializer=initializer))
    model.add(Dropout(0.4))
    model.add(Dense(1, kernel_initializer=initializer))
    model.compile(loss='mae', optimizer='adam', metrics=[coeff_deter])

    es = EarlyStopping(monitor='val_coeff_deter', mode='max', patience=5)

    model.fit(X_train, Y_train,
              batch_size=32,
              validation_data=(X_test, Y_test),
              epochs=50,
              callbacks=[es],
              verbose=2)

    return model


sp = pd.read_csv('sp500_stock.csv')
data = pd.read_csv('broader_stock.csv')

sp = data_preprocess(sp)
data = data_preprocess(data)
cutoff = int(data.shape[0] * 0.8)
observed_data = data.iloc[:cutoff]

ticker_list = list(data.columns)
ticker_list.remove('SPY')

tick = 'TGI'
arr = observed_data[tick]
'''
if len(arr) < 2000:
    continue
'''
coint_corr, coint_dta = coint_group(tick, observed_data)

# regression model

reg_model = regression_mod('%s_LAG' % tick, coint_dta)
y_pred = reg_model.predict()

# examine trading profit
regasset, regrecord = train_profit(tick, y_pred, coint_dta)

# machine learning model
train_length = 30
X = []
Y = []

for i in range(len(observed_data) - 120):
    x = observed_data[coint_corr].iloc[i:i + train_length].values.T.flatten()
    y = observed_data[tick].iloc[i + 120]
    X.append(x)
    Y.append(y)

X = np.array(X)
Y = np.array(Y).reshape(-1, 1)

print(X.shape, Y.shape)
mm_scaler_x = StandardScaler()
mm_scaler_y = StandardScaler()
lstm_mod = LSTM_mod(X, Y, mm_scaler_x, mm_scaler_y)

mm_scaler_x = mm_scaler_x.fit(X)
scale = mm_scaler_x.transform(X)
scale = scale.reshape(scale.shape[0], 1, scale.shape[1])

y_pred = lstm_mod.predict(scale)
mm_scaler_y = mm_scaler_y.fit(Y)
y_pred = mm_scaler_y.inverse_transform(y_pred)

# examine trading profit
mlasset, mlrecord = train_profit(tick, y_pred.flatten(), observed_data.shift(-30).iloc[:-30])
