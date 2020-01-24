import os
import yfinance as yf
from datetime import datetime
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


def data_collection(ticker_list, del_col):
    master = pd.DataFrame(columns=['Date'])
    for ticker in ticker_list:
        try:
            temp = yf.Ticker(ticker)
            hist_temp = temp.history(start=START, end=END)
            if len(hist_temp) >= 1500:
                hist_temp.drop(del_col, axis=1, inplace=True)
                hist_temp.columns = [ticker]
                master = pd.merge(master, hist_temp, on='Date', how='outer')
                status = "Successfully retrieve {}'s data.".format(ticker)
                print(status)
                logging.info(status)
            else:
                status = "Insufficient data for {}.".format(ticker)
                print(status)
                logging.info(status)
        except:
            status = "Failed to retrieve {}.".format(ticker)
            print(status)
            logging.info(status)
            pass
    cutoff = master[master.columns[2]].last_valid_index()
    master = master.iloc[:cutoff+1]
    return master


def preprocessing(df):
    del_ticker = []
    for ticker in df.columns[1:]:
        start = df[ticker].first_valid_index()
        dta = df[ticker].iloc[start:]
        if sum(dta.isna()) != 0:
            del_ticker.append(ticker)
    df.drop(del_ticker, axis=1, inplace=True)
    status = 'These tickers are dropped due to non-consecutive data series: {}'.format(del_ticker)
    print(status)
    logging.info(status)
    return df


if __name__ == '__main__':
    sp500_constituents = pd.read_csv("sp500_constituents.csv")
    sp500_list = sp500_constituents.loc[:, 'Symbol'].tolist()

    START = datetime(2001, 1, 1)
    END = datetime(2020, 1, 21)

    sp500_del_col = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits']

    init_logging()
    raw = data_collection(sp500_list, sp500_del_col)
    df = preprocessing(raw)
    df.to_csv('sp500.csv')
