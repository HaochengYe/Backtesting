# %%
import numpy as np
import pandas as pd
import cvxpy as cp

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
        covar[i] = data[ranking[i]].iloc[time+1-cycle:time+1]
    vol = np.array(covar.std(axis = 1))
    vol = np.reciprocal(vol)
    weight = vol / vol.sum()
    return weight    

rebalancing_strategies = [MinVariance, EqualWeight, MeanVariance_Constraint, RiskParity]
