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
    master.to_csv("sp500.csv")
    return None


if __name__ == '__main__':
    sp500_constituents = pd.read_csv("constituents_csv.csv")
    sp500_list = sp500_constituents.loc[:, 'Symbol'].tolist()

    START = datetime(2001, 1, 1)
    END = datetime(2019, 12, 25)

    sp500_del_col = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits']

    init_logging()
    data_collection(sp500_list, sp500_del_col)
