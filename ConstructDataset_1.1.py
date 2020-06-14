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


def data_collection(ticker_list, del_col, start, end):
    total = []
    for ticker in ticker_list:
        try:
            temp = yf.Ticker(ticker)
            hist_temp = temp.history(start=start, end=end)
            if hist_temp.shape[0] >= 500:
                hist_temp.drop(del_col, axis=1, inplace=True)
                hist_temp.columns = [ticker]
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


if __name__ == '__main__':
    txt_file = open('broader_ticker.txt', 'r')
    sp500_list = [line.rstrip('\n') for line in txt_file]

    START = datetime(1980, 1, 1)
    END = datetime(2020, 6, 14)

    sp500_del_col = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits']

    init_logging()
    raw = data_collection(sp500_list, sp500_del_col, START, END)
    df = pd.concat(raw, axis=1)

    df.dropna(how='all', inplace=True)
    df.to_csv('broader_stock.csv')
