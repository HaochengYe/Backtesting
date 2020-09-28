import matplotlib.pyplot as plt

import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

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
    y = dta['%s_LAG' % tick]
    cointegrat = {}
    correlat = {}

    for i in dta.columns[:-2]:
        x = dta[i]
        score, pval, _ = coint(x, y, trend='ct')
        corr = x.corr(y)

        cointegrat[i] = pval
        correlat[i] = corr

    best_coint = sorted(cointegrat, key=cointegrat.get)[:50]
    best_corr = sorted(correlat, key=correlat.get, reverse=True)[:50]

    intersect = list(set(best_coint) & set(best_corr))
    if len(intersect) > 0:
        print("There are {} cointegrated stocks.".format(len(intersect)))
        return intersect
    else:
        print("Intersection is empty.")
        return best_coint[:10]


def measure_profit(ture_val, fitted_val, asset):
    inventory = 0
    asset = asset
    record = [asset]

    for t in range(len(fitted_val)):
        trend_good = fitted_val[t] > ture_val[t]
        price = ture_val[t]
        if trend_good and inventory == 0:
            # buy
            asset -= price
            inventory += 1
        elif not trend_good and inventory == 1:
            # sell
            asset += price
            inventory -= 1
        elif t == len(fitted_val) - 1 and inventory == 1:
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
    X = dta[X].values
    Y = dta[Y].values
    mod = sm.OLS(Y, sm.add_constant(X)).fit()
    return mod


def l1_reg(X, Y, dta):
    X = dta[X].values
    Y = dta[Y].values
    mod = LassoCV(alphas=alphas, max_iter=5000, fit_intercept=True, cv=10, n_jobs=-1).fit(X, Y)
    return mod


def l2_reg(X, Y, dta):
    X = dta[X].values
    Y = dta[Y].values
    mod = RidgeCV(alphas=alphas, fit_intercept=True, cv=10).fit(X, Y)
    return mod


data = pd.read_csv('broader_stock.csv')
data = data_preprocess(data)

ticker_list = list(data.columns)
ticker_list.remove('SPY')

_ = int(0)
result = {}

alphas = np.linspace(0.001, 1000, 300)

for tick in ticker_list[0:500]:
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

    trading_data = original_data.iloc[int(original_data.shape[0] * 0.8):]

    original_data['%s_LAG' % tick] = original_data[tick].shift(-120)
    model_data = original_data.dropna()

    cutoff = int(model_data.shape[0] * 0.8)
    observed_data = model_data.iloc[:cutoff]
    test_data = model_data.iloc[cutoff:]

    arr = observed_data[tick]

    if len(arr) < 1000:
        _ += 1
        print("{} / {}".format(_, len(ticker_list)))
        continue

    coint_corr = coint_group(tick, observed_data)

    # This is the training period performance.
    # regression model
    reg_model = regression_mod(coint_corr, '%s_LAG' % tick, observed_data)
    l1_model = l1_reg(coint_corr, '%s_LAG' % tick, observed_data)
    l2_model = l2_reg(coint_corr, '%s_LAG' % tick, observed_data)

    # out-of sample mse
    X_test = test_data[coint_corr]
    Y_test = test_data["%s_LAG" % tick]

    l1_pred = l1_model.predict(X_test)
    l2_pred = l2_model.predict(X_test)
    ols_pred = reg_model.predict(sm.add_constant(X_test))

    l1_mse = mean_squared_error(Y_test.values, l1_pred)
    l2_mse = mean_squared_error(Y_test.values, l2_pred)
    ols_mse = mean_squared_error(Y_test.values, ols_pred.values)

    # selecting the best model
    mse_dict = {l1_mse: l1_model,
                l2_mse: l2_model,
                ols_mse: reg_model}

    best_model = mse_dict[min(mse_dict.keys())]

    y_trade = trading_data['%s' % tick].values
    x_trade = trading_data[coint_corr].values

    if min(mse_dict.keys()) == ols_mse:
        best_pred = best_model.predict(sm.add_constant(x_trade))
    else:
        best_pred = best_model.predict(x_trade)


    # examine trading profit
    init_asset = 0
    regasset, regrecord = measure_profit(y_trade, best_pred, init_asset)
    net_profit = (regasset - init_asset) - (y_trade[-1] - y_trade[0])
    var_in = np.var(regrecord) / len(regrecord)
    sharpe = net_profit / var_in

    result[tick] = [net_profit, regasset, var_in, sharpe, l1_mse, l2_mse, ols_mse]

    _ += 1
    print("{} / {}".format(_, len(ticker_list)))

result_dta = pd.DataFrame(result).T
result_dta.columns = ['NetProfit', 'GrossProfit', 'Var', 'Sharpe', 'L1_MSE', 'L2_MSE', 'OLS_MSE']
result_dta.to_csv('Regression_Prediction_1.csv')
