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


dta = pd.read_csv('broader_stock.csv')
dta = data_preprocess(dta)
pct_dta = dta.pct_change()

pred_res = pd.read_csv('XGB_Pred_1.csv')

target_list = list(set(pred_res['Unnamed: 0'][:10]) | {'TQQQ', 'UVXY'})
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
    temp_dta.dropna(inplace=True)

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
    opt_ft = pred_res[pred_res['Unnamed: 0'] == t].sort_values('sharpe', ascending=False)['Unnamed: 1'].iloc[0]

    X = temp_dta[feature_selection[opt_ft]].values
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
        'lambda': 1,
        'alpha': 1,
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

    final_val = xgb.DMatrix(pct_dta.iloc[-1][feature_selection[opt_ft]], label=[0])
    pred = model.predict(final_val)
    bin_pred = (pred > 0.5).astype(int)

    result[t] = [pred[0], bin_pred[0]]

print(result)
