# Backtesting
Factor investing backtesting platform


## Files:
ConstructDataset.py needs Python 2.7 to run
For those who wanna skip this step, I have also put the downloaded .csv file in this repository

Strategy.py has 5 trading strategies in it:
Moving Average
Price Reversal
Price Momentum
Momentum Return
Mean Cutoff
Price High - Price Low
Volatility Coefficient
Annualized Volatility
Moving Averages
MACD
Boolinger Bands

And 4 rebalancing strategies:
Minimizing Variance
Equal Weight
Risk Parity


It can also generate heatmaps to help visualize the return, volatility and Sharpe ratio given a specific trading and rebalancing combination.

## Things to improve:
1. Investigate the data source. NAs, monotony in the data that causes volatility to go to zero, how to handle dividing by zero in corner cases?
2. Find good trading strategies that involves short selling.
3. Backtesting rebalancing strategies that minimizes the volatility through short stocks with negative correlations

