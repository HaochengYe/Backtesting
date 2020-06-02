import copy

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Strategy import *

sns.set()

class Agent():

    def __init__(self, portfolio, data, trading_strategies, rebalancing_strategies, cycle, max_holding, gamma):

        self.portfolio = portfolio
        self.data = data
        self.trading_strategies = trading_strategies
        self.rebalancing_strategies = rebalancing_strategies
        self.cycle = cycle
        self.max_holding = max_holding

        # parameter for risk preference (proportion that invest into SPY)
        self.gamma = gamma

    @property
    def equity(self):
        return sum(self.portfolio.values())

    def PitchStock(self, strat, time):
        cycle = self.cycle
        data = self.data
        max_holding = self.max_holding
        ticker = data.columns.to_list()
        ranking = {}
        for i in ticker:
            metric = strat(data[i], cycle, time)
            if metric is not None and not math.isnan(metric):
                ranking[i] = metric
        result = sorted(ranking, key=ranking.get)[:max_holding]
        return result

    def Rebalance(self, ranking, strat, time):
        cycle = self.cycle
        data = self.data
        cash = self.portfolio['cash']
        # assume that cash earns risk-free rate interest
        equity = self.equity + cash * (self.rf - 1)
        target_portfolio = {}
        weight = np.array(strat(data, ranking, time, cycle))
        weight = (weight * equity).astype(int)
        for w, stock in zip(weight, ranking):
            price = data[stock].iloc[time]
            shares = w // price
            target_portfolio[stock] = shares
            equity -= shares * price
        target_portfolio['cash'] = equity
        return target_portfolio
