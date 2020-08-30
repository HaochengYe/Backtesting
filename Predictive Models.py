import matplotlib.pyplot as plt

import statsmodels.api as sm
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import coint
import warnings

from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


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

    for i in dta.columns[:-2]:
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


sp = pd.read_csv('sp500_stock.csv')
data = pd.read_csv('broader_stock.csv')

sp = data_preprocess(sp)
data = data_preprocess(data)

ticker_list = list(data.columns)
ticker_list.remove('SPY')

_ = int(0)
result = {}

for tick in ticker_list[0:100]:
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

    # This is the training period performance.
    # regression model
    reg_model = regression_mod(coint_corr, '%s_LAG' % tick, observed_data)
    y_pred = reg_model.predict()

    # examine trading profit
    regasset, regrecord = measure_profit(tick, y_pred, 0, observed_data)

    # Now we switch to actual testing where only fit models after 100 more new data points
    test_data = original_data.iloc[cutoff:]
    init_asset = 0
    regrecord_os = []

    # update every 100 new data
    T = test_data.shape[0] // 120

    for i in range(T):
        test_coint_corr = test_data[coint_corr].iloc[i * 120:(i + 1) * 120]
        y_pred_os = reg_model.predict(sm.add_constant(test_coint_corr))
        init_asset, record = measure_profit(tick, y_pred_os, init_asset, test_data.iloc[i * 120:(i + 1) * 120])
        regrecord_os += record

        # update model after record performance
        new_observed_data = original_data.iloc[i * 120:cutoff + (i + 1) * 120]
        coint_corr = coint_group(tick, new_observed_data)

        reg_model = regression_mod(coint_corr, '%s_LAG' % tick, new_observed_data)

    test_coint_corr = test_data[coint_corr].iloc[T * 120:]
    y_pred_os = reg_model.predict(sm.add_constant(test_coint_corr))
    regasset_os, record = measure_profit(tick, y_pred_os, init_asset, test_data.iloc[T * 120:])
    regrecord_os += record

    var_in = np.var(regrecord) / len(regrecord)
    var_os = np.var(regrecord_os) / len(regrecord_os)

    result[tick] = [regasset, var_in, regasset_os, var_os]

    _ += 1
    print("{} / {}".format(_, len(ticker_list)))

result_dta = pd.DataFrame(result)
result_dta.to_csv('Regression_Prediction.csv')
