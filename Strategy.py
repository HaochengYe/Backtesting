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

INITIAL_BALANCE = 10000
TRANS_COST = 0.01
# define the risk-free rate
RISKFREE = 1.02


# %%
def PriceReverse(df, cycle, time):
    """ Compute 1M Price Reversal as the following:
        PM_{i,t} = (Close_{i,t} - Close_{i, t-1}) / Close_{i, t-1}
        Order: Ascending
    Argument df: dataframe object (n*1 vector)
            cycle: how many days to look back to see its reversal
            time: current index for df to look at
    """
    try:
        previous_price = df.iloc[time - cycle]
        return (df.iloc[time] - previous_price) / previous_price
    except KeyError:
        return None

def Price_High_Low(df, cycle, time):
    """
    Compute High-minus-low:
    HL_{i,t} = (High_{i,t} - Close_{i,t}) / (Close_{i,t} - Low_{i,t})
    Order: Descending
    Argument df: dataframe object (n*1 vector)
            cycle: how many days to look back to see its reversal
            time: current index for df to look at
    """
    try:
        High = max(df.iloc[time-cycle:time])
        Low = min(df.iloc[time-cycle:time])
        return -(High - df.iloc[time]) / (df.iloc[time] - Low)
    except KeyError:
        return None

def Vol_Coefficient(df, cycle, time):
    """
    Compute Coefficient of Variation:
    CV_{i,t} = Std(Close_i, cycle) / Ave(Close_i, cycle)
    Order: Descending
        Argument df: dataframe object (n*1 vector)
            cycle: how many days to look back to see its reversal
            time: current index for df to look at
    """
    try:
        std = np.std(df.iloc[time-cycle:time])
        avg = np.mean(df.iloc[time-cycle:time])
        return -std / avg
    except KeyError:
        return None

def AnnVol(df, cycle, time):
    """
    Compute Coefficient of Variation:
    AnnVol = sqrt(252) * sqrt(1/21 * sum(r_{i,t-j}^2))
    where r_{i,s} = log(Close_{i,t} / Close_{i,t-1})
    Order: Descending
        Argument df: dataframe object (n*1 vector)
            cycle: how many days to look back to see its reversal
            time: current index for df to look at
    """
    try:
        r_2 = int(0)
        for i in range(1, cycle):
            log = np.log(df.iloc[time-i] / df.iloc[time-i-1])
            r_2 += log**2
        result = np.sqrt(252/cycle * r_2)
        return -result
    except KeyError:
        return None

strategies = [PriceReverse, Price_High_Low, Vol_Coefficient, AnnVol]


# %%
class Agent():
    def __init__(self, balance, data, strategies, cycle, max_holding):
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
        self.cycle = cycle
        self.equity = INITIAL_BALANCE
        self.re = float()
        self.tran_cost = float()
        self.rf = np.power(RISKFREE, self.cycle/252)
        self.max_holding = max_holding
        self.vol = float()


    def PitchStock(self, strategy, time):
        """
        Argument strategy: a function that takes (df, cycle, time) as argument
        return ranking: dictionary {Stock: Value} Value is some metric
        """
        cycle = self.cycle
        data = self.data
        max_holding = self.max_holding
        ranking = {}
        for i in ticker:
            ranking[i] = strategy(data[i], cycle, time)
        result = sorted(ranking, key = ranking.get)[:max_holding]
        return result

    def MeanVarWeight(self, ranking, time):
        """
        Argument ranking: list of stocks from PitchStock
                    return dictionary {Stock: Shares to buy}
        """
        cycle = self.cycle
        data = self.data
        covar = np.zeros((shape = ))
        for i in ranking:
            path = data[i].iloc[time+1-cycle:time+1]
        covar = np.concatenate((covar, path), axis = 1)
        
            

    def Trading(self, ranking, time):
        """
        Argument ranking: list of stocks
        returns nothing but changes the balance and record of the Agent
        """
        # take all necessary attributes from the class
        cost = 0
        equity = self.equity
        data = self.data
        balance = self.balance
        rf = self.rf
        max_holding = self.max_holding
        avail_cash = balance['cash'] * rf
        # buying
        for i in ranking:
            if i not in balance:
                num_to_buy = (equity / max_holding) // data[i].iloc[time]
                balance[i] = num_to_buy
                change = num_to_buy * data[i].iloc[time]
                cost += num_to_buy * TRANS_COST
                avail_cash -= change

        # selling
        for i in list(balance):
            if i not in ranking and i != 'cash':
                num_to_sell = balance[i]
                del balance[i]
                change = num_to_sell * data[i].iloc[time]
                cost += num_to_sell * TRANS_COST
                avail_cash += change

        # reassign values to the class attributes
        balance['cash'] = avail_cash
        self.balance = balance
        equity = equity + avail_cash - cost
        self.re = equity / INITIAL_BALANCE
        self.equity = equity
        self.tran_cost += cost
        
        
    def BackTesting(self):
        """
        This is backtsting for all strategies
        Return two dictionary
            1. return for each strategy
            2. overall cost for each strategy
        """
        cycle = self.cycle
        data = self.data
        strategies = self.strategies
        print("There are %s strategies we are testing." % len(strategies))
        print("They are: ")
        for i in strategies:
            print("     %s" % i.__name__)
        T = len(data) // cycle
        print("We are rebalancing for %s number of times." % T)
        portfolio_perform = {}
        for strategy in strategies:
            print("Testing %s" % strategy.__name__)
            for i in range(1, T):
                time = i * cycle
                ranking = self.PitchStock(strategy, time)
                self.Trading(ranking, time)
                print("Rebalancing for %s time!" % i)
            # compute the annualized return for this strategy
            result = np.power(self.re, 252 // cycle / T)
            portfolio_perform[strategy.__name__] = (result - 1)*100
            # reset balance, equity, re, and transaction cost for the agent
            self.reset()  
            print("\n")
        return portfolio_perform
    

    def BackTesting_Single(self, strategy):
        """
        This is backtsting for one single strategy
        Return two dictionary
            1. value path
            2. transaction cost path
        """
        cycle = self.cycle
        data = self.data
        print("Testing %s" % strategy.__name__)
        T = len(data) // cycle
        print("We are rebalancing for %s number of times." % T)
        portfolio_return = []
        for i in range(1, T):
            time = i * cycle
            ranking = self.PitchStock(strategy, time)
            self.Trading(ranking, time)
            print("Rebalancing for %s time!" % i)
            portfolio_return.append(self.equity)
        return portfolio_return

    
    def reset(self):
        """
        This reset the Agent to its initial holding. 
        Apply this method between testing different strategies.
        """
        self.balance = {'cash': INITIAL_BALANCE}
        self.equity = INITIAL_BALANCE
        self.re = float()
        self.tran_cost = float()
        self.vol = float()
            

# %%
def PitchStock(strategy, data, time):
        """
        Argument strategy: a function that takes (df, cycle, time) as argument
        return ranking: dictionary {Stock: Value} Value is some metric
        """
        ranking = {}
        for i in ticker:
            ranking[i] = strategy(data[i], 20, time)
        return ranking
        #result = sorted(ranking, key = ranking.get)[:20]
        #return result

# %%
wsw = Agent({'cash': INITIAL_BALANCE}, df[1250:], strategies, 5, 20)

# %%
ranking = wsw.PitchStock(strategies[0], 2476)
wsw.Trading(ranking, 2476)

# %%
wsw.BackTesting()

# %%