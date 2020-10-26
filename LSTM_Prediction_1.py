from sklearn.decomposition import PCA, FactorAnalysis

# ML imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import pandas as pd
import numpy as np
import random

from statsmodels.tsa.stattools import coint, adfuller


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


def measure_profit(price, indicator):
    assert len(price) == len(indicator)
    inventory = 0
    asset = 0
    record = [asset]
    for i in range(len(price)):
        trend_good = indicator[i] > 1
        p = price[i]
        if trend_good and inventory == 0:
            # buy
            asset -= p
            inventory += 1
        elif not trend_good and inventory == 1:
            # sell
            asset += p
            inventory -= 1
        elif i == len(price) - 1 and inventory == 1:
            asset += p
            inventory -= 1
        else:
            asset = record[-1]
        record.append(asset)
    return asset, record[1:]


def pct_change(arr):
    return np.diff(arr) / arr[1:]


def coint_group(tick, dta):
    """
    Use cointegration test and correlation to find predictive stocks for target
    :param tick: string for the target stock
    :param dta: the data file (csv) that contains the tick
    :return: a list of tickers that are in sp500 which predict the target
    """
    y = dta['%s_LAG' % tick]
    y = pct_change(y)
    cointegrat = {}
    correlat = {}

    for i in dta.columns[:-2]:
        x = dta[i]
        x = pct_change(x)
        score, pval, _ = coint(x, y, trend='ct')
        corr = x.corr(y)

        cointegrat[i] = pval
        correlat[i] = corr

    best_coint = sorted(cointegrat, key=cointegrat.get)[:50]
    best_corr = sorted(correlat, key=correlat.get, reverse=True)[:50]

    union = list(set(best_coint) | set(best_corr))
    union.append('SPY')
    return union


def chuck_data_transform(x, y, block_size):
    X_ml, Y_ml = [], []
    for i in range(x.shape[0] - block_size):
        x_temp = x[i:i+block_size,:].T.flatten()
        y_temp = y.iloc[i+block_size-5:i+block_size].values.T
        X_ml.append(x_temp)
        Y_ml.append(y_temp)

    X_ml = np.array(X_ml)
    X_ml = X_ml.reshape(X_ml.shape[0], 1, X_ml.shape[1])
    Y_ml = np.array(Y_ml)

    return X_ml, Y_ml


def LSTM_mod(x_ml, y_ml):
    xtr, xva, ytr, yva = train_test_split(x_ml, y_ml, test_size=0.2, random_state=42)

    initializer = initializers.glorot_normal(seed=42)
    model = Sequential()
    model.add(LSTM(50, input_shape=(xtr.shape[1], xtr.shape[2]), kernel_initializer=initializer, return_sequences=True))
    model.add(Dropout(0.2, seed=42))
    model.add(LSTM(20, input_shape=(xtr.shape[1], xtr.shape[2]), kernel_initializer=initializer))
    model.add(Dropout(0.2, seed=42))
    model.add(Dense(10, kernel_initializer=initializer))
    model.add(Dropout(0.2, seed=42))
    model.add(Dense(5, kernel_initializer=initializer))
    model.compile(loss='mae', optimizer='adam', metrics=[coeff_deter])
    es = EarlyStopping(monitor='val_coeff_deter', mode='max', patience=5)

    model.fit(xtr, ytr,
            batch_size=32,
            validation_data=(xva, yva),
            epochs=50,
            callbacks=[es],
            verbose=0)

    return model


# operation
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# set seed
np.random.seed(1)
tf.compat.v1.set_random_seed(1)
random.seed(1)

data = pd.read_csv('broader_stock.csv')
data = data_preprocess(data)

ticker_list = list(data.columns)
ticker_list.remove('SPY')

_ = int(0)
result = {}

for tick in ticker_list[500:1000]:
    original_series = data[tick]

    if tick in data.columns:
        original_data = pd.concat([data.drop([tick], axis=1), original_series], axis=1)
        original_data = original_data[original_data[tick].notnull()].dropna(axis=1)
    else:
        original_data = pd.concat([data, original_series], axis=1)
        original_data = original_data[original_data[tick].notnull()].dropna(axis=1)

    if original_data.index[-1] != data.index[-1]:
        _ += 1
        print("{} / {}".format(_, len(ticker_list)))
        continue

    original_data['%s_LAG' % tick] = original_data[tick].shift(-5)
    model_data = original_data.dropna()

    cutoff = int(model_data.shape[0] * 0.8)
    observed_data = model_data.iloc[:cutoff]
    test_data = model_data.iloc[cutoff:]

    # check data length
    arr = observed_data[tick]
    if len(arr) < 1000:
        _ += 1
        print("{} / {}".format(_, len(ticker_list)))
        continue
    
    coint_corr = coint_group(tick, observed_data)

    # transform data for LSTM
    X = observed_data[coint_corr]
    X = X.apply(pct_change, axis=0)
    Y = observed_data['%s_LAG' % tick]
    Y = pct_change(Y)

    X_test = test_data[coint_corr]
    X_test = X_test.apply(pct_change, axis=0)
    Y_test = test_data['%s_LAG' % tick]
    Y_test = pct_change(Y_test)

    # check stationarity
    adval = adfuller(Y.values)
    if adval[1] > 1e-4:
        _ += 1
        print("{} / {}".format(_, len(ticker_list)))
        continue

    # factor transformation
    transformer = FactorAnalysis(n_components=20, max_iter=5000, svd_method='lapack')
    X_transformed = transformer.fit_transform(X)
    X_test_tran = transformer.transform(X_test)

    block_size = 30
    X_ml, Y_ml = chuck_data_transform(X_transformed, Y, block_size)
    X_mltest, Y_mltest = chuck_data_transform(X_test_tran, Y_test, block_size)

    # LSTM fitting
    lstm_mod = LSTM_mod(X_ml, Y_ml)
    y_pred = lstm_mod.predict(X_mltest)
    fa_r2 = r2_score(Y_mltest, y_pred)

    # measuring profit
    indicator = np.apply_along_axis(lambda x: np.cumprod(1+x)[-1], 1, y_pred)
    true_price = model_data['%s_LAG' % tick].iloc[-len(indicator):].values
    ass, rec = measure_profit(true_price, indicator)

    ttl_ret = ass / true_price[0]
    net_ret = (ass - true_price[-1] + true_price[0]) / true_price[0]
    pct_record = np.array(rec) / np.array(true_price)
    var_record = np.var(pct_record)
    sharpe = net_ret / (var_record + 1e-10)

    # final prediction
    X_full = original_data[coint_corr]
    X_full = X_full.apply(pct_change, axis=0)
    X_full_tran = transformer.fit_transform(X_full)

    X_fullml = []
    for i in range(X_full_tran.shape[0] - block_size):
        x_temp = X_full_tran[i:i+block_size,:].T.flatten()
        X_fullml.append(x_temp)

    X_fullml = np.array(X_fullml)
    X_fullml = X_fullml.reshape(X_fullml.shape[0], 1, X_fullml.shape[1])

    t_pred = lstm_mod.predict(X_fullml[-1:,:,:])
    pred_ret = (1+t_pred).cumprod()[-1]

    result[tick] = [pred_ret, net_ret, ttl_ret, var_record, sharpe, fa_r2]

    _ += 1
    print("{} / {}".format(_, len(ticker_list)))

result_dta = pd.DataFrame(result).T
result_dta.columns = ['PredRet', 'NetProfit', 'GrossProfit', 'Var', 'Sharpe', 'R2']
result_dta.to_csv('LSTM_Prediction_1.csv')

