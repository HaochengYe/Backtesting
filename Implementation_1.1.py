import copy

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Strategy import *

sns.set()

class Agent:

    def __init__(self, data, trading_strategies, rebalancing_strategies, cycle, max_holding, gamma=0):

        self.portfolio = {'cash': INITIAL_BALANCE}
        self.data = data
        self.trading_strategies = trading_strategies
        self.rebalancing_strategies = rebalancing_strategies
        self.cycle = cycle
        self.max_holding = max_holding
        self.tran_cost = float(0)

        # parameter for risk preference (proportion that invest into SPY)
        self.gamma = gamma

    def get_equity(self, time):
        portfolio = copy.deepcopy(self.portfolio)
        cash = portfolio['cash']
        ttl_equity = cash

        del portfolio['cash']
        if len(portfolio.values()) != 0:
            shares = np.array(list(portfolio.values()))
            prices = np.array(self.data[list(portfolio.keys())].iloc[time])
            ttl_equity += shares @ prices
        return ttl_equity

    def PitchStock(self, trad_strat, rebal_strat, time):
        cycle = self.cycle
        data = self.data
        max_holding = self.max_holding
        equity = self.get_equity(time)
        ranking = {}
        for i in ticker:
            metric = trad_strat(data[i], cycle, time)
            if (metric is not None and not math.isnan(metric)) & (not math.isnan(data[i].iloc[time-cycle])):
                ranking[i] = metric
        result = sorted(ranking, key=ranking.get)[:max_holding]

        weight = np.array(rebal_strat(data, result, time, cycle))
        safe_weight = self.gamma
        target_weight = np.append(weight * (1 - safe_weight), safe_weight)
        dollar_weight = (target_weight * equity).astype(int)
        result.append('SPY')

        target_portfolio = {}
        for w, stock in zip(dollar_weight, result):
            price = data[stock].iloc[time]
            shares = w // price
            target_portfolio[stock] = shares
            equity -= shares * price

        target_portfolio['cash'] = equity

        return target_portfolio

    def Trade(self, target_portfolio, time):
        cost = float(0)
        data = self.data
        portfolio = self.portfolio

        for i in portfolio:
            if i not in target_portfolio and i != 'cash':
                cost += portfolio[i] * data[i].iloc[time] * TRANS_COST
            elif i in target_portfolio and i != 'cash':
                diff = abs(portfolio[i] - target_portfolio[i])
                cost += diff * data[i].iloc[time] * TRANS_COST

        for j in target_portfolio:
            if j not in portfolio:
                cost += target_portfolio[j] * data[j].iloc[time] * TRANS_COST

        self.tran_cost += cost
        self.portfolio = target_portfolio

    def Backtest_Single(self, trad_strat, rebal_strat):
        cycle = self.cycle
        data = self.data
        print("Trading strategy: %s \n" % trad_strat.__name__)
        print("Rebalancing strategy: %s \n" % rebal_strat.__name__)
        T = len(data) // cycle
        print("We are rebalancing for %s number of times." % T)

        portfolio_path = []
        _ = 0

        for t in range(1, len(data)):
            # non-trading period
            equity = self.get_equity(t)
            portfolio_path.append(equity)
            # trading period
            if t % cycle == 0:
                target_portfolio = self.PitchStock(trad_strat, rebal_strat, t)
                self.Trade(target_portfolio, t)
                _ += 1
                print("Rebalancing for %s time!" % _)
        return portfolio_path

    def reset(self):
        self.portfolio = {'cash': INITIAL_BALANCE}
        self.tran_cost = float(0)

    def Backtest_All(self):
        trading_strategies = self.trading_strategies
        rebalancing_strategies = self.rebalancing_strategies
        print("There are %s trading strategies and %s rebalancing strategies we are testing." % (
            len(trading_strategies), len(rebalancing_strategies)))
        print("Trading Strategies: ")
        for i in trading_strategies:
            print("     %s \n" % i.__name__)
        print("Rebalacing Strategies: ")
        for i in rebalancing_strategies:
            print("     %s \n" % i.__name__)

        portfolio_re = pd.DataFrame(index=[x.__name__ for x in rebalancing_strategies],
                                    columns=[x.__name__ for x in trading_strategies])
        portfolio_vol = pd.DataFrame(index=[x.__name__ for x in rebalancing_strategies],
                                     columns=[x.__name__ for x in trading_strategies])
        portfolio_sharpe = pd.DataFrame(index=[x.__name__ for x in rebalancing_strategies],
                                        columns=[x.__name__ for x in trading_strategies])

        for col, trad_strat in enumerate(trading_strategies):
            for row, rebal_strat in enumerate(rebalancing_strategies):
                path = self.Backtest_Single(trad_strat, rebal_strat)
                path = np.array(path)

                ttl_ret = np.maximum((path[-1] - self.tran_cost) / path[0], 0)
                annual_ret = np.power(np.power(ttl_ret, 1/len(path)), 252) - 1
                annual_vol = (np.diff(path) / path[1:]).std() * np.power(252, 1/2)
                annual_sharpe = (annual_ret - RISKFREE) / annual_vol

                portfolio_re.iloc[row][col] = annual_ret
                portfolio_vol.iloc[row][col] = annual_vol
                portfolio_sharpe.iloc[row][col] = annual_sharpe

                self.reset()
                print('\n')

        return portfolio_re, portfolio_vol, portfolio_sharpe

    def Backtest_History(self):
        trading_strategies = self.trading_strategies
        rebalancing_strategies = self.rebalancing_strategies
        print("There are %s trading strategies and %s rebalancing strategies we are testing." % (
            len(trading_strategies), len(rebalancing_strategies)))
        print("Trading Strategies: ")
        for i in trading_strategies:
            print("     %s \n" % i.__name__)
        print("Rebalacing Strategies: ")
        for i in rebalancing_strategies:
            print("     %s \n" % i.__name__)

        history = []
        cost = []

        for col, trad_strat in enumerate(trading_strategies):
            for row, rebal_strat in enumerate(rebalancing_strategies):
                path = self.Backtest_Single(trad_strat, rebal_strat)
                history.append(path)
                cost.append(self.tran_cost)
                self.reset()
                print('\n')

        return history, cost


def data_preprocess(dta):
    dta['Date'] = pd.to_datetime(dta['Date'], format='%Y-%m-%d')
    dta = dta.set_index(dta['Date'])
    # NHLI not traded
    dta.drop(['Date', 'NHLI'], axis=1, inplace=True)
    dta.dropna(how='all', inplace=True)
    for tick in dta.columns:
        tick_series = dta[tick]
        start_pos = tick_series.first_valid_index()
        valid_series = tick_series.loc[start_pos:]
        if valid_series.isna().sum() > 0:
            dta.drop(tick, axis=1, inplace=True)

    for tick in dta.columns:
        dta[tick] = dta[tick].mask(dta[tick] == 0).ffill(downcast='infer')

    return dta[dta.index >= dta['SPY'].first_valid_index()]


if __name__ == '__main__':
    df = pd.read_csv("broader_stock.csv")
    # df.drop(['Unnamed: 0'], axis=1, inplace=True)
    print(df.shape)
    df = data_preprocess(df)
    print(df.shape)
    ticker = list(df.columns)
    ticker.remove('SPY')

    INITIAL_BALANCE = 50000
    TRANS_COST = 0.001
    CYCLE = 5
    MAX_HOLDING = 30
    RISKFREE = 0.08325

    wsw = Agent(df[3000:], trading_strategies, rebalancing_strategies, CYCLE, MAX_HOLDING)
    # return_chart, vol_chart, sharpe_chart = wsw.Backtest_All()

    '''
    return_chart = return_chart.astype(float)
    plt.title('Return Heatmap')
    sns.heatmap(return_chart, annot=True, square=True, cmap='RdBu')

    vol_chart = vol_chart.astype(float)
    plt.title('Volatility Heatmap')
    sns.heatmap(vol_chart, annot=True, square=True, cmap='RdBu')

    sharpe_chart = sharpe_chart.astype(float)
    plt.title('Sharpe Ratio Heatmap')
    sns.heatmap(sharpe_chart, annot=True, square=True, cmap='RdBu')
    '''