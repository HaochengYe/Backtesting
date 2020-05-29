import copy

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Strategy import *

sns.set()

class Agent():

    def __init__(self, portfolio, trading_strategies, rebalancing_strategies, cycle, max_holding):

        self.portfolio = portfolio