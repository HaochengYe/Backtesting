# Backtesting
Factor investing backtesting platform


## Files:
ConstructDataset.py needs Python 2.7 to run
For those who wanna skip this step, I have also put the downloaded .csv file in this repository

Strategy.py has 5 trading strategies in it:
Price Reversal
Price Momentum
Price High - Price Low
Volatility Coefficient
Annualized Volatility

And 4 rebalancing strategies:
Minimizing Variance
Mean Variance with Constraints
Equal Weight
Risk Parity

Implementation.py includes the class object to:
1. Pitch Stock: generating a list of stocks to invest by one trading strategy
2. Rebalance: calculate the target portfolio weight by one rebalancing strategy
3. Trading: Adjust agent's portfolio to match the target portfolio
4. Backtest: which tests all combinations of trading and rebalancing strategies in Strategy.py

It can also generate heatmaps to help visualize the return, volatility and Sharpe ratio given a specific trading and rebalancing combination.

## Things to improve:
1. Add more strategies (trading or rebalancing) following the format in Strategy.py
2. Integration of strategies: Aggregate strategy performances by aggregating time relevance
3. Better source of data
4. Better visualization to illustrate the time relevance of these strategies.

