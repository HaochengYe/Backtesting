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
    dta['%s_LAG' % tick] = dta[tick].shift(-120)
    dta.dropna(inplace=True)

    y = dta['%s_LAG' % tick]
    cointegrat = {}
    correlat = {}

    for i in dta.columns[1:-2]:
        x = dta[i]
        score, pval, _ = coint(x, y, trend='ct')
        corr = x.corr(y)

        cointegrat[i] = pval
        correlat[i] = corr

    best_coint = sorted(cointegrat, key=cointegrat.get)[:10]
    best_corr = sorted(correlat, key=correlat.get, reverse=True)[:10]

    intersect = list(set(best_coint) & set(best_corr))
    if len(intersect) > 0:
        print("There are {} cointegrated stocks.".format(len(intersect)))
        return intersect
    else:
        print("Intersection is empty.")
        return best_coint[:3]


def measure_profit(tick, fitted_val, asset, dta):
    inventory = 0
    asset = asset
    record = [asset]
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

    return asset, record[1:]


def regression_mod(X, Y, dta):
    """
    Use basic regression model to forecast
    :param X: list of strings of tickers
    :param Y: string of lagged target ticker
    :param dta: the data set that contains X and Y
    :return: the regression model (statsmodels mod format)
    """
    X = dta[X]
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
              verbose=0)

    return model


sp = pd.read_csv('sp500_stock.csv')
data = pd.read_csv('broader_stock.csv')

sp = data_preprocess(sp)
data = data_preprocess(data)

ticker_list = list(data.columns)
ticker_list.remove('Date')
ticker_list.remove('SPY')

_ = int(0)
result = {}

for tick in ticker_list:

    original_series = data[tick]

    if tick in sp.columns:
        original_data = pd.concat([sp.drop([tick], axis=1), original_series], axis=1)
        original_data = original_data[original_data[tick].notnull()].dropna(axis=1)
    else:
        original_data = pd.concat([sp, original_series], axis=1)
        original_data = original_data[original_data[tick].notnull()].dropna(axis=1)

    if original_data.index[-1] != data.index[-1]:
        _ += 1
        print("{} / {}".format(_, len(ticker_list)))
        continue

    cutoff = int(original_data.shape[0] * 0.8)
    observed_data = original_data.iloc[:cutoff]

    arr = observed_data[tick]

    if len(arr) < 2000:
        _ += 1
        print("{} / {}".format(_, len(ticker_list)))
        continue

    coint_corr = coint_group(tick, observed_data)

    # machine learning model
    train_length = 30
    X, Y = [], []

    for i in range(len(observed_data) - train_length):
        x = observed_data[coint_corr].iloc[i:i + train_length].values.T.flatten()
        y = observed_data['%s_LAG' % tick].iloc[i + train_length]
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
    mlasset, mlrecord = measure_profit(tick, y_pred.flatten(), 0, observed_data.shift(-train_length).iloc[:-train_length])

    # Now we switch to actual testing where only fit models after 100 more new data points
    test_data = original_data.iloc[cutoff:]
    init_asset = 0
    mlrecord_os = []

    # update every 100 new data
    T = test_data.shape[0] // 120

    # machine learning model test data performance
    for i in range(T):

        test_X = []

        for t in range(120):
            test_prep = original_data[coint_corr].iloc[(cutoff-30+t+i*120):(cutoff+t+i*120)].values.T.flatten()
            test_X.append(test_prep)

        test_X = np.array(test_X)
        scale_test_X = mm_scaler_x.transform(test_X)
        scale_test_X = scale_test_X.reshape(scale_test_X.shape[0], 1, scale_test_X.shape[1])

        y_pred_os = lstm_mod.predict(scale_test_X)
        y_pred_os = mm_scaler_y.inverse_transform(y_pred_os)

        init_asset, record = measure_profit(tick, y_pred_os, init_asset, test_data.iloc[i * 120:(i + 1) * 120])
        mlrecord_os += record

        # update model after record performance
        new_observed_data = original_data.iloc[i * 120:cutoff + (i + 1) * 120]
        coint_corr = coint_group(tick, new_observed_data)

        X, Y = [], []

        for t in range(len(new_observed_data) - train_length):
            x = new_observed_data[coint_corr].iloc[t:t + train_length].values.T.flatten()
            y = new_observed_data['%s_LAG' % tick].iloc[t + train_length]
            X.append(x)
            Y.append(y)

        X = np.array(X)
        Y = np.array(Y).reshape(-1, 1)

        mm_scaler_x = StandardScaler()
        mm_scaler_y = StandardScaler()
        lstm_mod = LSTM_mod(X, Y, mm_scaler_x, mm_scaler_y)


    test_X = []

    for t in range(test_data.shape[0] % 120):
        test_prep = original_data[coint_corr].iloc[(cutoff-30+t+T*120):(cutoff+t+T*120)].values.T.flatten()
        test_X.append(test_prep)

    test_X = np.array(test_X)
    scale_test_X = mm_scaler_x.transform(test_X)
    scale_test_X = scale_test_X.reshape(scale_test_X.shape[0], 1, scale_test_X.shape[1])

    y_pred_os = lstm_mod.predict(scale_test_X)
    y_pred_os = mm_scaler_y.inverse_transform(y_pred_os)

    mlasset_os, record = measure_profit(tick, y_pred_os, init_asset, test_data.iloc[T * 120:])
    mlrecord_os += record

    # record information into dataframe
    var_in = np.var(mlrecord) / len(mlrecord)
    var_os = np.var(mlrecord_os) / len(mlrecord_os)

    result[tick] = [mlasset, var_in, mlasset_os, var_os]

    _ += 1
    print("{} / {}".format(_, len(ticker_list)))

result_dta = pd.DataFrame(result)
result_dta.to_csv('MachineLearning_Prediction.csv')
