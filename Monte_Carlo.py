import pandas as pd
import os
os.chdir(os.path.dirname(__file__))
#%%
import numpy as np
from PortfolioOpt import calc_optimal_weights_class_constraints
import sklearn.covariance
import sklearn.cluster
import matplotlib.pyplot as plt
#%%

data_dir = os.path.join(os.getcwd(), "Data")
output_dir = os.path.join(os.getcwd(), "Output")

tickers = []
adj_cl_df = pd.DataFrame()
if os.path.isdir(os.path.join(data_dir, 'stock_data')):
    for file in os.listdir(os.path.join(data_dir, 'stock_data')):
        if file.endswith(".csv"):
            ticker = file[:-4]
            df = pd.read_csv(os.path.join(data_dir, 'stock_data', file), index_col='Date')
            adj_cl_column = df['Adj Close'].rename(ticker)
            adj_cl_df = adj_cl_df.join(adj_cl_column, how = 'outer')
            tickers.append(ticker)

ret_df = adj_cl_df.pct_change().dropna()

correlation_with_spy = pd.read_csv(os.path.join(output_dir, 'correlation.csv'), index_col=0)
correlation_matrix= pd.read_csv(os.path.join(output_dir, 'corr.csv'), index_col=0)
corr2 = pd.read_csv(os.path.join(output_dir, 'correlation_squared.csv'),index_col=0)
betas = pd.read_csv(os.path.join(output_dir, 'Betas.csv'),index_col=0)
SPY_ret = 0.1
SPY_Stdev = 0.2
rf = 0.05
annual_ret = pd.DataFrame()
annual_ret = rf + betas*(SPY_ret-rf)
annual_var = corr2 * SPY_Stdev**2

daily_ret = annual_ret / 252
daily_var = annual_var / 252

std_dev_matrix = np.diag(np.sqrt(daily_var.iloc[:, 0])) 

corr_matrix = correlation_matrix.to_numpy()

daily_cov_matrix = np.dot(np.dot(std_dev_matrix, corr_matrix), std_dev_matrix)

daily_cov_df = pd.DataFrame(daily_cov_matrix, index=daily_var.index, columns=daily_var.index)
#%%
def user_input(SPY_annual_ret, SPY_annual_stdev, corr2, correlation_matrix, rf=0.05):
    annual_ret = pd.DataFrame()
    annual_ret = rf + betas*(SPY_annual_ret-rf)
    annual_var = corr2 * SPY_annual_stdev**2
    annual_var.rename(columns = {'SPY':''}, inplace = True)

    daily_ret = annual_ret / 252
    daily_var = annual_var / 252
    std_dev_matrix = np.diag(np.sqrt(daily_var.iloc[:, 0]))
    corr_matrix = correlation_matrix.to_numpy()
    daily_cov_matrix = np.dot(np.dot(std_dev_matrix, corr_matrix), std_dev_matrix)
    daily_cov_df = pd.DataFrame(daily_cov_matrix, index=daily_var.index, columns=daily_var.index)
    li = [daily_ret, daily_var, daily_cov_matrix]
    return li

#%%
asset_class_list = ['USCB', 'Commodity', 'EME', 'EMB', 'USE', 'USB']
ETF_li = ['QQQ', 'DIA', 'INDA', 'MCHI', 'TLT', 'LQD', 'HYG', 'DBC', 'EMB', 'VGSH']
asset_classes = {
    'VGSH': 'USB',
    'DBC': 'Commodity',
    'EMB': 'EMB',
    'QQQ': 'USE',
    'DIA': 'USE',
    'TLT': 'USB',
    'HYG': 'USCB',
    'INDA': 'EME',
    'MCHI': 'EME',
    'LQD':'USCB'
    }
class_constraints = {
    'USCB': {'min': 0.05, 'max': 0.8},
    'Commodity': {'min': 0.05, 'max': 0.8},
    'EME': {'min': 0.05, 'max': 0.8},
    'EMB': {'min': 0.05, 'max': 0.8},
    'USE': {'min': 0.05, 'max': 0.8},
    'USB': {'min': 0.05, 'max': 0.8}
}


def monte_carlo_simulation(type_, var_target_annual, confidence_level, SPY_annual_ret, SPY_annual_stdev, num_periods, num_simulations, hist = True):
    """
    Runs a Monte Carlo simulation on a portfolio.

    Args:
        type_ (str): Portfolio type.
        var_target_annual(float): Annual VaR target for the portfolio
        confidence_level(float): Confidence level for calculating VaR. NEW
        SPY_annual_ret(float): Expected returns for the market or SPY.
        SPY_annual_stdev(float): Estimated variance of the market or SPY.
        num_periods (int): Number of periods for the simulation.
        num_simulations (int): Number of simulation runs.
        hist(bool): Use historical data or expected return & standard deviation

    Returns:
        DataFrame: Simulated portfolio values.
    """
    
    asset_class_list = ['USCB', 'Commodity', 'EME', 'EMB', 'USE', 'USB']
    ETF_li = ['QQQ', 'DIA', 'INDA', 'MCHI', 'TLT', 'LQD', 'HYG', 'DBC', 'EMB', 'VGSH']
    asset_classes = {
        'VGSH': 'USB',
        'DBC': 'Commodity',
        'EMB': 'EMB',
        'QQQ': 'USE',
        'DIA': 'USE',
        'TLT': 'USB',
        'HYG': 'USCB',
        'INDA': 'EME',
        'MCHI': 'EME',
        'LQD':'USCB'
        }
    class_constraints = {
        'USCB': {'min': 0.05, 'max': 0.8},
        'Commodity': {'min': 0.05, 'max': 0.8},
        'EME': {'min': 0.05, 'max': 0.8},
        'EMB': {'min': 0.05, 'max': 0.8},
        'USE': {'min': 0.05, 'max': 0.8},
        'USB': {'min': 0.05, 'max': 0.8}
    }
    results = np.zeros((num_simulations, num_periods))
    weights = calc_optimal_weights_class_constraints(ret_df, asset_classes, class_constraints, objective=type_, var_target_annual=var_target_annual, confidence_level=confidence_level)
    if hist:
        covar = sklearn.covariance.ledoit_wolf(ret_df)[0]
        ret = ret_df.mean()
        var = ret_df.var()
    else:
        covar = user_input(SPY_annual_ret, SPY_annual_stdev, corr2, correlation_matrix, rf=0.05)[2]  
        ret = user_input(SPY_annual_ret, SPY_annual_stdev, corr2, correlation_matrix, rf=0.05)[0]  
        var = user_input(SPY_annual_ret, SPY_annual_stdev, corr2, correlation_matrix, rf=0.05)[1]  
    aligned_ret = ret.squeeze().reindex(weights.index).to_numpy()
    aligned_weights = weights.to_numpy()
    for i in range(num_simulations):
        random_returns = np.random.multivariate_normal(aligned_ret, covar, num_periods)
        portfolio_return = np.sum(random_returns * aligned_weights, axis=1)
        portfolio_value = (1 + portfolio_return).cumprod()
        results[i, :] = portfolio_value
    res = pd.DataFrame(results)
    return res



