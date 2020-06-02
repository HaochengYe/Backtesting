import copy

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Strategy import *

sns.set()


INIT_BALANCE = 50000
TRAN_COST = 0.05


class Agent:

    def __init__(self, data, trading_strategies, rebalancing_strategies, cycle, max_holding, gamma):

        self.portfolio = {}
        self.data = data
        self.trading_strategies = trading_strategies
        self.rebalancing_strategies = rebalancing_strategies
        self.cycle = cycle
        self.max_holding = max_holding

        # parameter for risk preference (proportion that invest into SPY)
        self.gamma = gamma

    def get_equity(self, time):
        portfolio = self.portfolio
        shares = np.array(portfolio.values())
        prices = np.array(self.data[list(portfolio.keys())].iloc[time])
        self.equity = shares @ prices
        return self.equity

    def PitchStock(self, strat, time):
        cycle = self.cycle
        data = self.data
        max_holding = self.max_holding
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
        equity = self.get_equity(time)
        weight = np.array(strat(data, ranking, time, cycle))
        safe_weight = self.gamma

        target_weight = np.append(weight * (1 - safe_weight), safe_weight)
        target_posit = target_weight * equity

        target_portfolio = dict(zip(ranking.append('SPY'), target_weight))




