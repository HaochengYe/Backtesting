import math

import cvxpy as cp
import numpy as np


# %%
def PriceReverse(df, cycle, time):
    """
    Compute 1M Price Reversal
    Order: Ascending
    :param df: dataframe object (n*1 vector)
    :param cycle: how many days to look back to see its reversal
    :param time: current index for df to look at
    :return: PM_{i,t} = (Close_{i,t} - Close_{i, t-1}) / Close_{i, t-1}
    """
    try:
        previous_price = df.iloc[time - cycle]
        return (df.iloc[time] - previous_price) / previous_price
    except KeyError:
        pass


def PriceMomentum(df, cycle, time):
    """
    Compute 1M Price Reversal
    Order: Descending
    :param df: dataframe object (n*1 vector)
    :param cycle: how many days to look back to see its reversal
    :param time: current index for df to look at
    :return: PM_{i,t} = (Close_{i,t} - Close_{i, t-1}) / Close_{i, t-1}
    """
    try:
        previous_price = df.iloc[time - 2*cycle]
        return -(df.iloc[time] - previous_price) / previous_price
    except KeyError:
        pass


def Price_High_Low(df, cycle, time):
    """
    Compute High-minus-low:
    Order: Descending
    :param df: dataframe object (n*1 vector)
    :param cycle: how many days to look back to see its reversal
    :param time: current index for df to look at
    :return: HL_{i,t} = (High_{i,t} - Close_{i,t}) / (Close_{i,t} - Low_{i,t})
    """
    try:
        High = max(df.iloc[time - cycle:time])
        Low = min(df.iloc[time - cycle:time])
        return -(High - df.iloc[time]) / (df.iloc[time] - Low)
    except KeyError:
        pass


def Vol_Coefficient(df, cycle, time):
    """
    Compute Coefficient of Variation:
    Order: Descending
    :param df: dataframe object (n*1 vector)
    :param cycle: how many days to look back to see its reversal
    :param time: current index for df to look at
    :return: CV_{i,t} = Std(Close_i, cycle) / Ave(Close_i, cycle)
    """
    try:
        data = df.iloc[time - cycle:time].pct_change(1).dropna()
        std = np.std(data)
        avg = np.mean(data)
        return -std / avg
    except KeyError:
        pass


def AnnVol(df, cycle, time):
    """
    Compute Annual Volatility:
    Order: Descending
    :param df: dataframe object (n*1 vector)
    :param cycle: how many days to look back to see its reversal
    :param time: current index for df to look at
    :return: AnnVol = sqrt(252) * sqrt(1/21 * sum(r_{i,t-j}^2))
    where r_{i,s} = log(Close_{i,t} / Close_{i,t-1})
    """
    try:
        r_2 = int(0)
        for i in range(1, cycle):
            log = np.log(df.iloc[time - i] / df.iloc[time - i - 1])
            r_2 += log ** 2
        result = np.sqrt(252 / cycle * r_2)
        return -result
    except KeyError:
        pass


def MovingAverage(df, cycle, time):
    """
    Compute Moving Average:
    Order: Descending
    :param df: dataframe object (n*1 vector)
    :param cycle: how many days to look back to see its reversal
    :param time: current index for df to look at
    :return: (MA_10 - Price) + (MA_20 - Price) * 2 + (MA_50 - Price) * 5
    """
    try:
        data = df.iloc[time - cycle:time]
        MA_10 = list(data.rolling(10).mean())[-1]
        MA_20 = list(data.rolling(20).mean())[-1]
        MA_50 = list(data.rolling(50).mean())[-1]
        res = MA_10 + MA_20 + MA_50 - df.iloc[time]*3
        return res
    except KeyError:
        pass


def MACD(df, cycle, time):
    """
    Compute Moving Average Convergence Divergence:
    Order: Descending
    :param df: dataframe object (n*1 vector)
    :param cycle: how many days to look back to see its reversal
    :param time: current index for df to look at
    :return: cycle-Period EMA - cycle*2-Period EMA
    where EMA = Price_t * k + EMA_t-1 * (1-k)
    k = 2 / (N+1)
    """
    try:
        data = df.iloc[time - cycle:time]
        EMA_SR = data.ewm(span=cycle).mean()
        EMA_LR = data.ewm(span=cycle*2).mean()
        res = list(EMA_SR)[-1] - list(EMA_LR)[-1]
        return res
    except KeyError:
        pass


def BoolingerBands(df, cycle, time):
    """
    Compute Boolinger Bands:
    Order: Descending
    :param df: dataframe object (n*1 vector)
    :param cycle: how many days to look back to see its reversal
    :param time: current index for df to look at
    :return: Ave(cycle) +- 2 * Std(cycle)
    """
    if time - 2 * cycle <= 0 and not math.isnan(df.iloc[time - cycle]):
        return 0
    try:
        data_lr = df.iloc[time - 2*cycle+1:time]
        data_sr = df.iloc[time - cycle:time]
        SMA = data_lr.rolling(cycle).mean()
        SMA.dropna(inplace=True)
        delta = np.std(data_sr)
        up_bound = SMA + delta
        lw_bound = SMA - delta
        midpoint = len(data_sr) // 2
        res = sum(data_sr[:midpoint] > up_bound[:midpoint]) - sum(data_sr[midpoint:] < lw_bound[midpoint:])
        return res * np.std(data_sr.pct_change(1).dropna())
    except ValueError:
        pass



trading_strategies = [PriceReverse, PriceMomentum, Price_High_Low, Vol_Coefficient, AnnVol, MACD, BoolingerBands]


def MinVariance(data, ranking, time, cycle):
    """
    MinVariance minimizes variance (needs short positions)
    Argument ranking: list of stocks from PitchStock
            return weighting for each stock (in percentage)
    """
    covar = np.zeros(shape=(len(ranking), cycle))
    for i in range(len(ranking)):
        covar[i] = data[ranking[i]].iloc[time-cycle:time].fillna(method='Backfill')
    inv_cov_matrix = np.linalg.pinv(np.cov(covar))
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
    covar = np.zeros(shape=(len(ranking), cycle))
    for i in range(len(ranking)):
        covar[i] = data[ranking[i]].iloc[time-cycle:time].fillna(method='Backfill')
    cov_matrix = np.cov(covar)
    weight = cp.Variable(shape=len(ranking))
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
    covar = np.zeros(shape=(len(ranking), cycle))
    for i in range(len(ranking)):
        covar[i] = data[ranking[i]].iloc[time - cycle:time]
    vol = np.array(covar.std(axis=1))
    vol = np.reciprocal(vol)
    weight = vol / vol.sum()
    return weight


rebalancing_strategies = [MinVariance, EqualWeight, MeanVariance_Constraint, RiskParity]
