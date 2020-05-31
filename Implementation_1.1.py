import copy

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Strategy import *

sns.set()

class Agent():

    def __init__(self, portfolio, data, trading_strategies, rebalancing_strategies, cycle, max_holding):

        self.portfolio = portfolio
        self.data = data
        self.trading_strategies = trading_strategies
        self.rebalancing_strategies = rebalancing_strategies
        self.cycle = cycle


    @property
    def equity(self):
        