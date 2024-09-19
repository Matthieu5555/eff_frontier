import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import minimize
from tqdm import tqdm  # For the loading bar
import plotly.express as px

#!!! things to improve

def download_data(tickers, period):
    """
    Downloads adjusted close prices for the given tickers and time period using yfinance.

    Parameters:
    - tickers (list): List of ticker symbols.
    - period (str): Time period for data download (e.g., '2y', '1y', '6mo', 'max').

    Returns:
    - prices (DataFrame): DataFrame of adjusted close prices.
    """
    prices = yf.download(tickers, period=period)['Adj Close']
    prices.dropna(inplace=True)
    return prices

def calculate_returns(prices):
    """
    Calculates daily returns from adjusted close prices.

    Parameters:
    - prices (DataFrame): DataFrame of adjusted close prices.

    Returns:
    - daily_returns (DataFrame): DataFrame of daily returns.
    """
    daily_returns = prices.pct_change().dropna()
    return daily_returns

def calculate_average_returns(tickers_list, period):
    """
    Calculates the average return series of the specified tickers.

    Parameters:
    - tickers_list (list): List of ticker symbols.
    - period (str): Time period for data download.

    Returns:
    - average_returns (Series): Series representing the combined asset.
    """
    prices = yf.download(tickers_list, period=period)['Adj Close']
    prices.dropna(inplace=True)
    daily_returns = prices.pct_change().dropna()
    average_returns = daily_returns.mean(axis=1)
    return average_returns

def annualize_returns(daily_returns):
    """
    Calculates annualized expected returns and covariance matrix from daily returns.

    Parameters:
    - daily_returns (DataFrame): DataFrame of daily returns.

    Returns:
    - expected_returns (Series): Annualized expected returns.
    - covariance_matrix (DataFrame): Annualized covariance matrix.
    """
    expected_returns = daily_returns.mean() * 252  # There are approximately 252 trading days in a year
    covariance_matrix = daily_returns.cov() * 252
    return expected_returns, covariance_matrix

def calculate_risk_free_rate(ticker, period):
    """
    Calculates the risk-free rate using the average annualized return of the specified ticker.

    Parameters:
    - ticker (str): Ticker symbol for the risk-free asset.
    - period (str): Time period for data download.

    Returns:
    - risk_free_rate (float): Annualized expected return of the risk-free asset.
    """
    prices = yf.download(ticker, period=period)['Adj Close']
    prices.dropna(inplace=True)
    returns = prices.pct_change().dropna()
    risk_free_rate = returns.mean() * 252  # Annualized expected return
    return risk_free_rate

def portfolio_return(weights_variable, expected_returns, fixed_weights, variable_tickers):
    """
    Calculates the expected return of a portfolio including fixed weights.

    Parameters:
    - weights_variable (array): Weights of variable assets.
    - expected_returns (Series): Expected returns of all assets.
    - fixed_weights (dict): Fixed weights for certain assets.
    - variable_tickers (list): List of variable ticker symbols.

    Returns:
    - portfolio_return (float): Expected portfolio return.
    """
    # Combine variable and fixed weights
    weights = pd.Series(weights_variable, index=variable_tickers)
    weights = pd.concat([weights, pd.Series(fixed_weights)])
    weights = weights[expected_returns.index]  # Ensure correct order
    return np.dot(weights, expected_returns)

def portfolio_volatility(weights_variable, covariance_matrix, fixed_weights, variable_tickers):
    """
    Calculates the volatility (standard deviation) of a portfolio including fixed weights.

    Parameters:
    - weights_variable (array): Weights of variable assets.
    - covariance_matrix (DataFrame): Covariance matrix of asset returns.
    - fixed_weights (dict): Fixed weights for certain assets.
    - variable_tickers (list): List of variable ticker symbols.

    Returns:
    - portfolio_volatility (float): Portfolio volatility.
    """
    # Combine variable and fixed weights
    weights = pd.Series(weights_variable, index=variable_tickers)
    weights = pd.concat([weights, pd.Series(fixed_weights)])
    weights = weights[covariance_matrix.columns]  # Ensure correct order
    return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

def minimize_volatility(target_return, expected_returns, covariance_matrix, fixed_weights, variable_tickers, total_fixed_weight):
    """
    Minimizes portfolio volatility for a given target return, including fixed weights.

    Parameters:
    - target_return (float): Target expected return for the portfolio.
    - expected_returns (Series): Expected returns of all assets.
    - covariance_matrix (DataFrame): Covariance matrix of asset returns.
    - fixed_weights (dict): Fixed weights for certain assets.
    - variable_tickers (list): List of variable ticker symbols.
    - total_fixed_weight (float): Sum of fixed weights.

    Returns:
    - optimal_weights (array): Optimal weights for variable assets.
    """
    n = len(variable_tickers)
    initial_guess = np.repeat((1 - total_fixed_weight) / n, n)
    bounds = ((0.0, 1.0),) * n
    # Constraints
    weights_sum_to_target = {
        'type': 'eq',
        'fun': lambda weights_variable: np.sum(weights_variable) - (1 - total_fixed_weight)
    }
    return_is_target = {
        'type': 'eq',
        'args': (expected_returns, fixed_weights, variable_tickers),
        'fun': lambda weights_variable, expected_returns, fixed_weights, variable_tickers: portfolio_return(
            weights_variable, expected_returns, fixed_weights, variable_tickers) - target_return
    }
    result = minimize(portfolio_volatility, initial_guess, args=(covariance_matrix, fixed_weights, variable_tickers),
                      method='SLSQP', bounds=bounds,
                      constraints=(weights_sum_to_target, return_is_target))
    return result.x

def maximum_sharpe_ratio(risk_free_rate, expected_returns, covariance_matrix, fixed_weights, variable_tickers, total_fixed_weight):
    """
    Maximizes the Sharpe Ratio of the portfolio, including fixed weights.

    Parameters:
    - risk_free_rate (float): Risk-free rate for Sharpe Ratio calculation.
    - expected_returns (Series): Expected returns of all assets.
    - covariance_matrix (DataFrame): Covariance matrix of asset returns.
    - fixed_weights (dict): Fixed weights for certain assets.
    - variable_tickers (list): List of variable ticker symbols.
    - total_fixed_weight (float): Sum of fixed weights.

    Returns:
    - optimal_weights (array): Optimal weights for variable assets.
    """
    n = len(variable_tickers)
    initial_guess = np.repeat((1 - total_fixed_weight) / n, n)
    bounds = ((0.0, 1.0),) * n
    # Constraints
    weights_sum_to_target = {
        'type': 'eq',
        'fun': lambda weights_variable: np.sum(weights_variable) - (1 - total_fixed_weight)
    }
    # Objective function to maximize Sharpe Ratio
    def negative_sharpe_ratio(weights_variable, risk_free_rate, expected_returns, covariance_matrix, fixed_weights, variable_tickers):
        port_return = portfolio_return(weights_variable, expected_returns, fixed_weights, variable_tickers)
        port_volatility = portfolio_volatility(weights_variable, covariance_matrix, fixed_weights, variable_tickers)
        return -(port_return - risk_free_rate) / port_volatility
    result = minimize(negative_sharpe_ratio, initial_guess, args=(risk_free_rate, expected_returns, covariance_matrix, fixed_weights, variable_tickers),
                      method='SLSQP', bounds=bounds,
                      constraints=(weights_sum_to_target,))
    return result.x

def global_minimum_volatility(covariance_matrix, fixed_weights, variable_tickers, total_fixed_weight):
    """
    Finds the portfolio with the global minimum volatility, including fixed weights.

    Parameters:
    - covariance_matrix (DataFrame): Covariance matrix of asset returns.
    - fixed_weights (dict): Fixed weights for certain assets.
    - variable_tickers (list): List of variable ticker symbols.
    - total_fixed_weight (float): Sum of fixed weights.

    Returns:
    - optimal_weights (array): Optimal weights for variable assets.
    """
    n = len(variable_tickers)
    initial_guess = np.repeat((1 - total_fixed_weight) / n, n)
    bounds = ((0.0, 1.0),) * n
    # Constraints
    weights_sum_to_target = {
        'type': 'eq',
        'fun': lambda weights_variable: np.sum(weights_variable) - (1 - total_fixed_weight)
    }
    result = minimize(portfolio_volatility, initial_guess, args=(covariance_matrix, fixed_weights, variable_tickers),
                      method='SLSQP', bounds=bounds,
                      constraints=(weights_sum_to_target,))
    return result.x

def compute_efficient_frontier(num_points, expected_returns, covariance_matrix, fixed_weights, variable_tickers, total_fixed_weight):
    """
    Computes the efficient frontier by calculating optimal portfolios for a range of target returns.

    Parameters:
    - num_points (int): Number of portfolios to compute along the efficient frontier.
    - expected_returns (Series): Expected returns of all assets.
    - covariance_matrix (DataFrame): Covariance matrix of asset returns.
    - fixed_weights (dict): Fixed weights for certain assets.
    - variable_tickers (list): List of variable ticker symbols.
    - total_fixed_weight (float): Sum of fixed weights.

    Returns:
    - efficient_frontier_df (DataFrame): DataFrame containing weights, returns, and volatilities of the portfolios.
    """
    target_returns = np.linspace(expected_returns.min(), expected_returns.max(), num_points)
    weights_list = []
    returns_list = []
    volatilities_list = []
    for target_return in tqdm(target_returns, desc="Calculating optimal weights"):
        try:
            optimal_weights = minimize_volatility(target_return, expected_returns, covariance_matrix, fixed_weights, variable_tickers, total_fixed_weight)
            # Combine variable and fixed weights
            full_weights = pd.Series(optimal_weights, index=variable_tickers)
            full_weights = pd.concat([full_weights, pd.Series(fixed_weights)])
            full_weights = full_weights[expected_returns.index]  # Ensure correct order
            weights_list.append(full_weights.values)
            portfolio_ret = np.dot(full_weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(full_weights.T, np.dot(covariance_matrix, full_weights)))
            returns_list.append(portfolio_ret)
            volatilities_list.append(portfolio_vol)
        except Exception as e:
            print(f"An error occurred for target return {target_return}: {e}")
    efficient_frontier_df = pd.DataFrame(weights_list, columns=expected_returns.index)
    efficient_frontier_df['Returns'] = returns_list
    efficient_frontier_df['Volatility'] = volatilities_list
    return efficient_frontier_df

def plot_efficient_frontier(num_points, expected_returns, covariance_matrix, fixed_weights, variable_tickers, total_fixed_weight, risk_free_rate):
    """
    Plots the efficient frontier using Plotly, including options to show the capital market line,
    equal weight portfolio, and global minimum volatility portfolio.

    Parameters:
    - num_points (int): Number of portfolios to compute along the efficient frontier.
    - expected_returns (Series): Expected returns of all assets.
    - covariance_matrix (DataFrame): Covariance matrix of asset returns.
    - fixed_weights (dict): Fixed weights for certain assets.
    - variable_tickers (list): List of variable ticker symbols.
    - total_fixed_weight (float): Sum of fixed weights.
    - risk_free_rate (float): Risk-free rate for Sharpe Ratio calculation.

    Returns:
    - fig (Figure): Plotly Figure object with the plot.
    - efficient_frontier_df (DataFrame): DataFrame containing the efficient frontier data.
    """
    ef_df = compute_efficient_frontier(num_points, expected_returns, covariance_matrix, fixed_weights, variable_tickers, total_fixed_weight)
    
    fig = go.Figure()

    # Plot the efficient frontier
    fig.add_trace(go.Scatter(
        x=ef_df['Volatility'],
        y=ef_df['Returns'],
        mode='lines+markers',
        name='Efficient Frontier',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))

    # Plot the capital market line
    optimal_weights_variable = maximum_sharpe_ratio(risk_free_rate, expected_returns, covariance_matrix, fixed_weights, variable_tickers, total_fixed_weight)
    full_weights_msr = pd.Series(optimal_weights_variable, index=variable_tickers)
    full_weights_msr = pd.concat([full_weights_msr, pd.Series(fixed_weights)])
    full_weights_msr = full_weights_msr[expected_returns.index]  # Ensure correct order
    msr_return = np.dot(full_weights_msr, expected_returns)
    msr_volatility = np.sqrt(np.dot(full_weights_msr.T, np.dot(covariance_matrix, full_weights_msr)))
    cml_x = [0, msr_volatility]
    cml_y = [risk_free_rate, msr_return]
    fig.add_trace(go.Scatter(
        x=cml_x,
        y=cml_y,
        mode='lines',
        name='Capital Market Line',
        line=dict(color='green', width=2, dash='dash')
    ))

    # Plot the equal weight portfolio
    n = len(variable_tickers)
    weights_ew_variable = np.repeat((1 - total_fixed_weight) / n, n)
    full_weights_ew = pd.Series(weights_ew_variable, index=variable_tickers)
    full_weights_ew = pd.concat([full_weights_ew, pd.Series(fixed_weights)])
    full_weights_ew = full_weights_ew[expected_returns.index]  # Ensure correct order
    ew_return = np.dot(full_weights_ew, expected_returns)
    ew_volatility = np.sqrt(np.dot(full_weights_ew.T, np.dot(covariance_matrix, full_weights_ew)))
    fig.add_trace(go.Scatter(
        x=[ew_volatility],
        y=[ew_return],
        mode='markers',
        name='Equal Weight Portfolio',
        marker=dict(color='goldenrod', size=10, symbol='star')
    ))

    # Plot the global minimum volatility portfolio
    optimal_weights_gmv = global_minimum_volatility(covariance_matrix, fixed_weights, variable_tickers, total_fixed_weight)
    full_weights_gmv = pd.Series(optimal_weights_gmv, index=variable_tickers)
    full_weights_gmv = pd.concat([full_weights_gmv, pd.Series(fixed_weights)])
    full_weights_gmv = full_weights_gmv[expected_returns.index]  # Ensure correct order
    gmv_return = np.dot(full_weights_gmv, expected_returns)
    gmv_volatility = np.sqrt(np.dot(full_weights_gmv.T, np.dot(covariance_matrix, full_weights_gmv)))
    fig.add_trace(go.Scatter(
        x=[gmv_volatility],
        y=[gmv_return],
        mode='markers',
        name='Global Minimum Volatility Portfolio',
        marker=dict(color='midnightblue', size=10, symbol='diamond')
    ))

    fig.update_layout(
        title='Efficient Frontier with Fixed Weights',
        xaxis_title='Volatility (Standard Deviation)',
        yaxis_title='Expected Return',
        legend_title='Portfolios',
        hovermode='closest'
    )

    fig.update_xaxes(tickformat='.2%')
    fig.update_yaxes(tickformat='.2%')

    fig.show()
    return fig, ef_df



def prepare_portfolio_data(tickers, real_estate_tickers, time_period):
    """
    Prepares the portfolio data by downloading stock prices, calculating returns,
    and handling the real estate average separately.
    
    Parameters:
    - tickers (list): List of stock tickers.
    - real_estate_tickers (list): List of real estate tickers to be averaged.
    - time_period (str): Time period for data download.
    
    Returns:
    - daily_returns (DataFrame): Daily returns for all assets including real estate average.
    - expected_returns (Series): Annualized expected returns.
    - covariance_matrix (DataFrame): Annualized covariance matrix.
    """
    # Download data for all tickers except real estate ones
    all_tickers = tickers + ['^FCHI']
    prices = download_data(all_tickers, time_period)
    
    # Calculate daily returns
    daily_returns = calculate_returns(prices)
    
    # Calculate average returns for the real estate tickers
    avg_real_estate_returns = calculate_average_returns(real_estate_tickers, time_period)
    
    # Add the average real estate returns to the daily_returns DataFrame
    daily_returns['RealEstate_Avg'] = avg_real_estate_returns
    
    # Calculate expected returns and covariance matrix
    expected_returns, covariance_matrix = annualize_returns(daily_returns)
    
    return daily_returns, expected_returns, covariance_matrix

def load_tickers_from_csv(file_path):
    """
    Loads tickers from a CSV file.
    """
    df = pd.read_csv(file_path)
    # Ensure the required column 'Ticker' is present
    if 'Ticker' not in df.columns:
        raise ValueError("CSV file must contain a 'Ticker' column.")
    tickers = df['Ticker'].dropna().tolist()
    return tickers

def main():
    csv_file_path = input("Enter the path to the CSV file containing tickers: ")
    tickers = load_tickers_from_csv(csv_file_path)

    real_estate_tickers = ["GFC.PA", "COV.PA"]
    
    time_period = input("Enter the time period for data download (e.g., '2y', '1y', '6mo', 'max'): ")
    
    # Prepare portfolio data
    daily_returns, expected_returns, covariance_matrix = prepare_portfolio_data(tickers, real_estate_tickers, time_period)
    
    # Define fixed weights !!!
    fixed_weights = {'^FCHI': 0.05, 'RealEstate_Avg': 0.65, 'GLD': 0.01, 'GBTC': 0.01}
    total_fixed_weight = sum(fixed_weights.values())
    
    # Update the list of variable tickers
    variable_tickers = [t for t in daily_returns.columns if t not in fixed_weights.keys()]
    
    # Calculate risk-free rate !!!
    risk_free_rate = calculate_risk_free_rate('OBLI.PA', time_period)
    
    # Plot the Efficient Frontier !!!
    num_points = 50
    fig, ef_df = plot_efficient_frontier(num_points, expected_returns, covariance_matrix, fixed_weights, variable_tickers, total_fixed_weight, risk_free_rate)
    
    # Calculate optimal weights
    optimal_weights_variable = maximum_sharpe_ratio(risk_free_rate, expected_returns, covariance_matrix, fixed_weights, variable_tickers, total_fixed_weight)
    
    # Combine variable and fixed weights
    optimal_weights_full = pd.Series(optimal_weights_variable, index=variable_tickers)
    optimal_weights_full = pd.concat([optimal_weights_full, pd.Series(fixed_weights)])
    optimal_weights_full = optimal_weights_full[expected_returns.index]
    
    # Prepare and print optimal weights
    optimal_weights_df = pd.DataFrame({
        'Ticker': optimal_weights_full.index,
        'Weight': optimal_weights_full.values
    })
    optimal_weights_df = optimal_weights_df[optimal_weights_df['Weight'] > 0.0001]
    optimal_weights_df['Weight (%)'] = optimal_weights_df['Weight'] * 100
    optimal_weights_df = optimal_weights_df.sort_values(by='Weight', ascending=False)
    print("\nOptimal Portfolio Weights:")
    print(optimal_weights_df[['Ticker', 'Weight (%)']].reset_index(drop=True))
    
    # Plot pie chart
    fig_pie = px.pie(optimal_weights_df, values='Weight', names='Ticker', title='Optimal Portfolio Allocation')
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.show()

if __name__ == "__main__":
    main()
