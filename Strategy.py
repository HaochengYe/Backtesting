# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("Dataset/SP500.csv")

# %%
ticker = df.columns
# rebalance portfolio every month (20 trading days)
REBALANCE_PERIOD = 20
INITIAL_BALANCE = 1e4
MAX_HOLDING_NUM = 20

# %%
def PriceReverse(df, cycle, time):
    """ Compute 1M Price Reversal as the following:
        PM_{i,t} = (Close_{i,t} - Close_{i, t-1}) / Close_{i, t-1}
    Argument df: dataframe object (n*1 vector)
            cycle: how many days to look back to see its reversal
            time: current index for df to look at
    """
    try:
        previous_price = df.loc[time - cycle]
        return (df.loc[time] - previous_price) / previous_price
    except KeyError:
        return None

def Price_High_Low(df, cycle, time):
    """
    Compute High-minus-low:
    HL_{i,t} = (High_{i,t} - Close_{i,t}) / (Close_{i,t} - Low_{i,t})
    Argument df: dataframe object (n*1 vector)
            cycle: how many days to look back to see its reversal
            time: current index for df to look at
    """
    try:
        High = max(df.loc[time-cycle:time])
        Low = min(df.loc[time-cycle:time])
        return (High - df.loc[time]) / (df.loc[time] - Low)
    except KeyError:
        return None

def Vol_Coefficient(df, cycle, time):
    """
    Compute Coefficient of Variation:
    CV_{i,t} = Std(Close_i, cycle) / Ave(Close_i, cycle)
        Argument df: dataframe object (n*1 vector)
            cycle: how many days to look back to see its reversal
            time: current index for df to look at
    """
    try:
        std = np.std(df.loc[time-cycle:time])
        avg = np.mean(df.loc[time-cycle:time])
        return std / avg
    except KeyError:
        return None

def AnnVol(df, cycle, time):
    """
    Compute Coefficient of Variation:
    AnnVol = sqrt(252) * sqrt(1/21 * sum(r_{i,t-j}^2))
    where r_{i,s} = log(Close_{i,t} / Close_{i,t-1})
        Argument df: dataframe object (n*1 vector)
            cycle: how many days to look back to see its reversal
            time: current index for df to look at
    """
    try:
        r_2 = int(0)
        for i in range(1, cycle):
            log = np.log(df.loc[i] / df.loc[i-1])
            r_2 += log**2
        result = np.sqrt(252/cycle * r_2)
        return result
    except KeyError:
        return None

# %%

test = pd.DataFrame(df.groupby(axis=1).apply(PriceReverse,10, 20))
    
    
    

# %%

def execute(ranking):
    """
    Argument ranking: dictionary {Stock_Ticker: Value} Value is one of the metrics
    return: dictionary {Stock_Ticker: Purchase Amount}
    """
    to_purchase = 
