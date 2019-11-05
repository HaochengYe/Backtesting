# Backtesting
Factor investing backtesting platform

ConstructDataset.py needs Python 2.7 to run
For those who wanna skip this step, I have also put the downloaded .csv file in this repository

Strategy.py has factor strategies in it.

Improvements:
Two directions (hyperparameter tuning & integration of strategies)
1. Parameters that are subjectively choose are: max_holding_num, cycle. And variables that should be observed from the market but now are personally assigned are: transaction_cost, risk_free_rate. We can find optimization methods that choose these parameters in an optimal way under different scenarios given these environment variables. Perhaps even we could draw an efficient frontier since we only have two parameters that need to be optimized.
2. All strategies should exhibit some seasonalities for a given macroeconomic condition of the market. We could delve further into the correlation between the strategies, and develop a new function of switching strategies based on some signals.