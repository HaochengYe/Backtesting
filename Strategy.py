# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
df = pd.read_csv("SP500.csv")
df.drop(['Unnamed: 0'], axis = 1, inplace=True)
print(df.shape)
ticker = list(df.columns)[1:]
# rebalance portfolio every month (20 trading days)
cycle = 20
INITIAL_BALANCE = 1e4
MAX_HOLDING_NUM = 20
TRANS_COST = 0.01

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

strategies = [PriceReverse, Price_High_Low, Vol_Coefficient, AnnVol]

# %%



# %%
class Agent():
    def __init__(self, balance, data, strategies):
        """
        Balance: dictionary (accounting book)
        Max_holding is the maximum number of stocks this agent can hold
        Cycle is the rebalancing period
        Data is the dataset
        Strategies is which factor investing stratgy this Agent has in disposal
        """
        self.balance = balance
        self.data = data
        self.strategies = strategies


    def PitchStock(self, strategy, time):
        """
        Argument strategy: a function that takes (df, cycle, time) as argument
        return ranking: dictionary {Stock: Value} Value is some metric
        """
        data = self.data
        ranking = {}
        for i in ticker:
            ranking[i] = strategy(data[i], cycle, time)
        result = sorted(ranking, key = ranking.get)[:20]
        return result
            

    def Trading(self, ranking, time):
        """
        Argument ranking: list of stocks
                record: dictionary (accounting book for holdings)
        returns nothing but changes the balance and record of the Agent
        """
        data = self.data
        balance = self.balance
        avail_cash = balance['cash']
        for i in ranking:
            if i not in balance:
                num_to_buy = (INITIAL_BALANCE / MAX_HOLDING_NUM) // data[i][time]
                balance[i] = num_to_buy
                avail_cash -= num_to_buy * data[i][time]
        for i in list(balance):
            if i not in ranking and i != 'cash':
                num_to_sell = balance[i]
                del balance[i]
                avail_cash += num_to_sell * data[i][time]
        balance['cash'] = avail_cash
        self.balance = balance
        
        
    def BackTesting(self):
        """
        Return a dictionary {Strat1: Return, Strat2: Return...}
        """
        T = len(self.data) // cycle
        

# %%
def PitchStock(strategy, data, time):
        """
        Argument strategy: a function that takes (df, cycle, time) as argument
        return ranking: dictionary {Stock: Value} Value is some metric
        """
        ranking = {}
        for i in ticker:
            ranking[i] = strategy(data[i], cycle, time)
        return ranking
        # result = sorted(ranking, key = ranking.get)[:20]
        # return result

# %%
# testing environment
wsw = Agent({'cash': INITIAL_BALANCE}, test_data, strategies)
ranking = wsw.PitchStock(strategies[0], 40)
wsw.Trading(ranking, 40)