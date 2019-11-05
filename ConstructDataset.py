# %%
# need to run in python 2.7
import sys
sys.path.append('D:/GitHub/finsymbols')
from finsymbols import symbols
import os
import pandas_datareader as pdr
from datetime import datetime
import pandas as pd
import numpy as np

# %%
# get the ticker for the S&P 500 company
sp500 = symbols.get_sp500_symbols()
ticker_list = []
# change dictionary to a list of ticker_list
for i in range(len(sp500)):
    unicode = sp500[i]['symbol']
    ticker_list.append(str(unicode).strip('\n'))

START = datetime(2010,1,1)
END = datetime(2019, 11, 4)
# only keeps "adj close" column, delete the rest
del_col = ['High', 'Low', 'Open', 'Close', 'Volume']
# master dataset where contain all S&P 500 data
master = pd.DataFrame(columns=['Date'])


# %%
%%time
for ticker in ticker_list:
    try:
        dataset = pdr.get_data_yahoo(symbols = ticker, start = START, end = END)
        # only take data with more than 100 months for testing (as the strategy will rebalance every month)
        if len(dataset) > 2000:
            dataset.drop(del_col, axis = 1, inplace = True)
            # rename the individual stock closing price as its ticker
            dataset.columns = [ticker]
            master = pd.merge(master, dataset, how = 'outer', on = 'Date')
        print(ticker)
    except KeyError:
        print(ticker + " failed!")
        pass

# %%
master.to_csv('SP500.csv')