import os
import yfinance as yf
from datetime import datetime, date
import pandas as pd
import pytz
import logging


def init_logging():
    global tz
    tz = pytz.timezone('US/Eastern')
    if not os.path.exists('log'):
        os.makedirs('log')

    date = datetime.now(tz).date().strftime('%Y%m%d')
    logging.basicConfig(filename='log/{}.log'.format(date), level=logging.INFO)

    return None


def data_collection(ticker_list, del_col, start, end):
    total = []
    for ticker in ticker_list:
        try:
            temp = yf.Ticker(ticker)
            hist_temp = temp.history(start=start, end=end)
            if hist_temp.shape[0] >= 500:
                hist_temp.drop(del_col, axis=1, inplace=True)
                hist_temp = hist_temp.rename(columns={'Open': ticker + '_Open', 'Close': ticker + '_Close', 'Volume': ticker + '_Volume'})
                status = "Successfully retrieve {}'s data.".format(ticker)
                print(status)
                logging.info(status)
                total.append(hist_temp)
            else:
                status = "Insufficient data for {}.".format(ticker)
                print(status)
                logging.info(status)
        except:
            status = "Failed to retrieve {}.".format(ticker)
            print(status)
            logging.info(status)
            pass
    return total


def data_preprocess(dta):
    # dta['Date'] = pd.to_datetime(dta['Date'], format='%Y-%m-%d')
    # dta = dta.set_index(dta['Date'])
    # NHLI not traded
    # dta.drop(['Date', 'NHLI'], axis=1, inplace=True)
    # dta.dropna(how='all', inplace=True)
    for tick in dta.columns:
        tick_series = dta[tick]
        start_pos = tick_series.first_valid_index()
        valid_series = tick_series.loc[start_pos:]
        if valid_series.isna().sum() > 1:
            dta.drop(tick, axis=1, inplace=True)

    for tick in dta.columns:
        dta[tick] = dta[tick].mask(dta[tick] == 0).ffill(downcast='infer')

    return dta[dta.index >= dta['SPY_Close'].first_valid_index()]


if __name__ == '__main__':
    txt_file = open('broader_ticker.txt', 'r')
    sp500_list = [line.rstrip('\n') for line in txt_file]

    START = datetime(2000, 1, 1)
    END = date.today()

    sp500_del_col = ['High', 'Low', 'Dividends', 'Stock Splits']

    init_logging()
    raw = data_collection(sp500_list, sp500_del_col, START, END)
    df = pd.concat(raw, axis=1)

    df.dropna(how='all', inplace=True)
    df = data_preprocess(df)
    df.to_csv('broader_stock.csv')
