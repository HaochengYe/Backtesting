import yfinance as yf
from datetime import datetime, date
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

START = datetime(2000, 1, 1)
END = date.today()

dta = pd.read_csv('broader_stock.csv')


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


dta = data_preprocess(dta)
pct_dta = dta.pct_change()

target_list = dta.columns.to_list() + ['TQQQ', 'UVXY']
mod_list = [0, 1, 2]
idx = 1

result = {}
for t in target_list:
    print("-----------{}/{}-----------".format(idx, len(target_list)))
    idx += 1
    target_tick = yf.Ticker(t)
    hist = target_tick.history(start=START, end=END)
    pct_target = hist.pct_change()

    if t in pct_dta.columns:
        temp_dta = pd.concat([pct_target.Close, pct_dta.drop([t], axis=1)], axis=1)
    else:
        temp_dta = pd.concat([pct_target.Close, pct_dta], axis=1)

    temp_dta = temp_dta[temp_dta['Close'].notnull()]
    temp_dta['Close_LAG'] = temp_dta['Close'].shift(-1)
    temp_dta = temp_dta.iloc[:-1].dropna(axis=1)

    cointegrat = {}
    correlat = {}

    for col in temp_dta.columns[:-1]:
        x = temp_dta[col]
        score, pval, _ = coint(x, temp_dta['Close_LAG'], autolag='t-stat')
        corr = abs(x.corr(temp_dta['Close_LAG']))
        cointegrat[col] = pval
        correlat[col] = corr

    best_coint = sorted(cointegrat, key=cointegrat.get)[:50]
    best_corr = sorted(correlat, key=correlat.get, reverse=True)[:50]
    union_X = list(set(best_coint) | set(best_corr))

    feature_selection = [best_coint, best_corr, union_X]
    for m in mod_list:
        X = temp_dta[feature_selection[m]].values
        y = temp_dta['Close_LAG'].values
        y = (y > 0).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42,
                                                            stratify=y)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            # Parameters that we are going to tune.
            'max_depth': 2,
            'min_child_weight': 1,
            'eta': .3,
            'subsample': 1,
            'colsample_bytree': 1,
            # Other parameters
            'objective': 'binary:logistic',
            'eval_metric': 'auc'
        }

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtest, "Test")],
            early_stopping_rounds=10
        )

        y_pred = model.predict(dtest)
        y_pred = (y_pred > 0.5).astype(int)
        test_acc = accuracy_score(y_test, y_pred)

        ttl_dtest = xgb.DMatrix(X, np.zeros(X.shape[0]))
        ttl_pred = model.predict(ttl_dtest)
        ttl_pred = (ttl_pred > 0.5).astype(int)

        inventory = 0
        asset = 0
        record = [asset]

        for i, dt in enumerate(temp_dta[union_X].index):
            price = hist.loc[dt]['Close']
            trend_good = ttl_pred[i] == 1
            if trend_good and inventory == 0 and i != len(ttl_pred) - 1:
                # buy
                asset -= price
                inventory += 1
            elif not trend_good and inventory == 1:
                # sell
                asset += price
                inventory -= 1
            elif i == len(ttl_pred) - 1 and inventory == 1:
                # liquidate in the end
                asset += price
                inventory -= 1
            else:
                asset = record[-1]
            record.append(asset)

        sub_hist = hist.loc[temp_dta[union_X].index]

        ttl_ret = asset / sub_hist.iloc[0].Close
        net_ret = (asset - sub_hist.iloc[-1].Close + sub_hist.iloc[0].Close) / sub_hist.iloc[0].Close
        pct_record = np.array(record[1:]) / np.array(sub_hist.Close)
        sharpe = net_ret / np.var(pct_record) + 1e-10

        # last day prediction
        final_val = xgb.DMatrix(pct_dta.iloc[-1][feature_selection[m]], label=[0])
        pred = (model.predict(final_val) > 0.5).astype(int)

        result[t, m] = [ttl_ret, net_ret, sharpe, test_acc, pred]

pd_res = pd.DataFrame(result).T
pd_res.columns = ['ttl_ret', 'net_ret', 'sharpe', 'acc', 'pred']
pd_res.to_csv('XGB_Pred_1.csv')