# eff_frontier

## Description:

This Python program was developed to solve an asset management case, aimed at optimizing portfolio allocation while incorporating real estate exposure and fixed weight constraints. The tool allows users to input a list of stock tickers, along with fixed weights for certain assets, and calculates the optimal portfolio weights by maximizing the Sharpe ratio or minimizing volatility. The program fetches historical data from Yahoo Finance, computes daily returns, annualizes expected returns and covariances, and builds an efficient frontier. It provides visualizations including the efficient frontier, Capital Market Line, and portfolio allocations.

## Key Features:

Flexible stock selection via CSV input
Integration of fixed asset weights (e.g., real estate) into portfolio construction
Efficient frontier calculation with adjustable target returns
Optimal weight determination via minimization of volatility or maximization of Sharpe ratio
Visual outputs: Efficient frontier plot, Capital Market Line, and portfolio allocation pie chart
Handling of real estate assets as an average of selected tickers
Files:

The program is contained in a single Python script and uses common libraries including yfinance, pandas, numpy, Plotly, and SciPy. It also includes a CSV loader for tickers.
