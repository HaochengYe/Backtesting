# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
import math

# %%
df = pd.read_csv("SP500.csv")
df.drop(['Unnamed: 0'], axis = 1, inplace=True)
print(df.shape)
ticker = list(df.columns)[1:]
# rebalance portfolio every month (20 trading days)

INITIAL_BALANCE = 24628
TRANS_COST = 0.05
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

def PriceMomentum(df, cycle, time):
    """ Compute 1M Price Momentum as the following:
        PM_{i,t} = (Close_{i,t} - Close_{i, t-1}) / Close_{i, t-1}
        Order: Descending
    Argument df: dataframe object (n*1 vector)
            cycle: how many days to look back to see its reversal
            time: current index for df to look at
    """
    try:
        previous_price = df.iloc[time - cycle]
        return -(df.iloc[time] - previous_price) / previous_price
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

trading_strategies = [PriceReverse, PriceMomentum, Price_High_Low, Vol_Coefficient, AnnVol]

# %%
def MinVariance(data, ranking, time, cycle):
    """
    MinVariance minimizes variance (needs short positions)
    Argument ranking: list of stocks from PitchStock
            return weighting for each stock (in percentage)
    """
    covar = np.zeros(shape = (len(ranking), cycle))
    for i in range(len(ranking)):
        covar[i] = data[ranking[i]].iloc[time+1-cycle:time+1]
    inv_cov_matrix = np.linalg.inv(np.cov(covar))
    ita = np.ones(inv_cov_matrix.shape[0])
    weight = (inv_cov_matrix @ ita) / (ita @ inv_cov_matrix @ ita)
    return weight

def EqualWeight(data, ranking, time, cycle):
    """
    EqualWeight assign weight by 1/N
    return weighting for each stock (in percentage)
    """
    N = len(ranking)
    weight = np.ones(shape=N) / N
    return weight

def MeanVariance_Constraint(data, ranking, time, cycle):
    """
    Mean Variance solved by convex optimization
    return weighting for each stock (in percentageg)
    """
    covar = np.zeros(shape = (len(ranking), cycle))
    for i in range(len(ranking)):
        covar[i] = data[ranking[i]].iloc[time+1-cycle:time+1]
    cov_matrix = np.cov(covar)
    weight = cp.Variable(shape = len(ranking))
    objective = cp.Minimize(cp.quad_form(weight, cov_matrix))
    constraints = [cp.sum(weight) == 1, weight >= 1 / (2 * len(ranking))]
    problem = cp.Problem(objective, constraints)
    result = problem.solve()
    return weight.value


def RiskParity(data, ranking, time, cycle):
    """
    RiskParity inversely invest for stock according to their volatility
    disregards covariance is the major drawback
    return weighting for each stock (in percentage)
    """
    covar = np.zeros(shape = (len(ranking), cycle))
    for i in range(len(ranking)):
        covar[i] = df[ranking[i]].iloc[2000+1-cycle:2000+1]
    vol = covar.std(axis = 1)
    weight = vol / vol.sum()
    return weight    


rebalancing_strategies = [MinVariance, EqualWeight, MeanVariance_Constraint, RiskParity]

# %%
class Agent():
    def __init__(self, portfolio, data, trade_strategies, rebalancing_strategies, cycle, max_holding):
        """
        portfolio: dictionary (accounting book)
        Max_holding is the maximum number of stocks this agent can hold
        Cycle is the rebalancing period
        Data is the dataset
        Strategies is which factor investing stratgy this Agent has in disposal
        """
        self.portfolio = portfolio
        self.data = data
        self.trade_strategies = trade_strategies
        self.rebalancing_strategies = rebalancing_strategies
        self.cycle = cycle
        self.equity = INITIAL_BALANCE
        self.re = float()
        self.tran_cost = float()
        self.rf = np.power(RISKFREE, self.cycle/252)
        self.max_holding = max_holding


    def PitchStock(self, trading_strategy, time):
        """
        Argument trading_strategy: a function that takes (df, cycle, time) as argument
        return ranking: list of stocks that should invest
        """
        cycle = self.cycle
        data = self.data
        max_holding = self.max_holding
        ranking = {}
        for i in ticker:
            ranking[i] = trading_strategy(data[i], cycle, time)
        result = sorted(ranking, key = ranking.get)[:max_holding]
        return result


    def Rebalancing(self, ranking, rebalancing_strategy, time):
        """
        Argument ranking: result from Agent.PitchStock
                rebalancing_strategy: a function that takes (df, ranking, time, cycle) as argument
                return target_portfolio: dictionary {Stock: # of shares}
        """
        cycle = self.cycle
        data = self.data
        cash = self.portfolio['cash']
        # assume that cash earns risk-free rate interest
        equity = self.get_Equity(time) + cash * (self.rf - 1)
        target_portfolio = {}
        weight = np.array(rebalancing_strategy(data, ranking, time, cycle))
        weight = (weight * equity).astype(int)
        for w, stock in zip(weight, ranking):
            price = data[stock].iloc[time]
            shares = w // price
            target_portfolio[stock] = shares
            equity -= shares * price
        target_portfolio['cash'] = equity
        return target_portfolio


    def get_Equity(self, time):
        """
        return the equity value for a given time
            sum weight * price
        """
        data = self.data
        portfolio = self.portfolio
        cash = portfolio['cash']
        ticker = list(portfolio)
        ticker.remove('cash')
        total_equity = cash
        for stock in ticker:
            price = data[stock].iloc[time]
            total_equity += price * portfolio[stock]
        return total_equity
        

    def Trading(self, target_portfolio, time):
        """
        Argument target_portfolio: a dictionary get from rebalance 
                    (what Agent.portfolio should be after trading)
        returns nothing but update:
                equity, portfolio, re, tran_cost
        """
        # take all necessary attributes from the class
        cost = 0
        portfolio = self.portfolio
        # selling and adjust share
        for i in list(portfolio):
            if i not in target_portfolio and i != 'cash':
                cost += portfolio[i] * TRANS_COST
            elif i in target_portfolio and i != 'cash':
                diff = abs(portfolio[i] - target_portfolio[i])
                cost += diff * TRANS_COST
        # buying
        for i in target_portfolio:
            if i not in portfolio:
                cost += target_portfolio[i] * TRANS_COST
        # update all the attribute of the agent
        self.tran_cost += cost
        self.portfolio = target_portfolio
        self.equity = self.get_Equity(time) - cost
        self.re = self.equity / INITIAL_BALANCE
        
            
    def BackTesting(self):
        """
        This is backtsting for all strategies
        Return two dictionary
            1. return for each strategy
            2. overall cost for each strategy
        """
        trading_strategies = self.trading_strategies
        rebalancing_strategies = self.rebalancing_strategies
        print("There are %s trading strategies and %s rebalancing strategies we are testing." % (len(trading_strategies), len(rebalancing_strategies)))
        print("They are: ")
        for i in trading_strategies:
            print("     %s" % i.__name__)
        print('\n')
        for i in rebalancing_strategies:
            print("     %s" % i.__name__)
        portfolio_re = pd.DataFrame(columns=range(len(trading_strategies)), index = range(len(rebalancing_strategies)))
        portfolio_vol = pd.DataFrame(columns=range(len(trading_strategies)), index = range(len(rebalancing_strategies)))
        portfolio_sharpe = pd.DataFrame(columns=range(len(trading_strategies)), index = range(len(rebalancing_strategies)))
        for trading_strategy in trading_strategies:
            for rebalancing_strategy in rebalancing_strategies:
                # use BackTesting_Single to get the three value of metrics needed
                total_return, vol, sharpe = self.BackTesting_Single(trading_strategy, rebalancing_strategy)

            portfolio_perform[strategy.__name__] = [total_return, vol, sharpe]
            # reset balance, equity, re, and transaction cost for the agent
            self.reset()
            print("\n")
        # turn this dictionary into a nicely presentable dataframe
        table = pd.DataFrame.from_dict(portfolio_perform, orient='index')
        table.columns = ['Annualized Return', 'Volatility', 'Sharpe Ratio']
        return table
    

    def BackTesting_Single(self, trading_strategy, rebalancing_strategy):
        """
        This is backtsting for one single combination of trading and rebalancing strategy
        Return the total return, volatility and Sharpe ratio
        """
        cycle = self.cycle
        data = self.data
        print("Trading strategy: %s" % trading_strategy.__name__)
        print("\n")
        print("Rebalancing strategy: %s" % rebalancing_strategy.__name__)
        T = len(data) // cycle
        print("We are rebalancing for %s number of times." % T)
        portfolio_re = []
        for i in range(1, T):
            time = i * cycle
            ranking = self.PitchStock(trading_strategy, time)
            target_portfolio = self.Rebalancing(ranking, rebalancing_strategy, time)
            self.Trading(target_portfolio, time)
            print("Rebalancing for %s time!" % i)
            portfolio_re.append(self.re)
        vol = np.std(portfolio_re)
        total_return = (np.power(self.re, 252 // cycle / T) - 1)*100
        sharpe = (total_return - (self.rf - 1)*100) / vol
        return total_return , vol, sharpe


    def reset(self):
        """
        This reset the Agent to its initial holding. 
        Apply this method between testing different strategies.
        """
        self.portfolio = {'cash': INITIAL_BALANCE}
        self.equity = INITIAL_BALANCE
        self.re = float()
        self.tran_cost = float()
            

# %%


# %%
wsw = Agent({'cash': INITIAL_BALANCE}, df, trading_strategies, rebalancing_strategies, 20, 10)

# %%
ranking = wsw.PitchStock(trading_strategies[0], 20)
target = wsw.Rebalancing(ranking, rebalancing_strategies[3], 20)
wsw.Trading(target, 20)

# %%
wsw.BackTesting()

# %%
wsw.BackTesting_Single(PriceMomentum, RiskParity)

# %%