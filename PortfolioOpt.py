import pandas as pd
import os
import numpy as np
import sklearn.cluster
import sklearn.covariance
import sklearn.manifold
from scipy.optimize import minimize



os.chdir(os.path.dirname(__file__))
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

#%%
def calc_optimal_weights_class_constraints(
    returns, 
    asset_classes, 
    class_constraints, 
    weight_bounds=(0.01, 1.0), 
    rf=0.045, 
    covar_method="ledoit-wolf",
    objective='VaR_MaxReturn',
    var_target_annual=None,
    confidence_level=0.95,
    options={'maxiter': 10000}
):
    """
    NOTE: THIS FUNCTION IS A MODIFICATION OF 'ffn' library - calc_mean_var_weights. Please visit https://pmorissette.github.io/ffn/_modules/ffn/core.html#calc_mean_var_weights for more details. 

    Args:
        * returns (DataFrame): Returns for multiple securities.
        * asset_classes (dict): Mapping of asset to asset class. NEW
        * class_constraints (dict): Constraints for each asset class. NEW
        * weight_bounds ((low, high)): Weight limits for each asset. This is a default input in ffn, we can just use (0, 1). THIS IS NOT ASSET ALLOCATION LIMIT! 
        * rf (float): Risk-free rate used in utility calculation.
        * covar_method (str): Covariance matrix estimation method.
        * objective (str): Objective of optimal portfolio. NEW
        * var_target_annual: Annual VaR target for the portfolio. NEW
        * confidence_level: Confidence level for calculating VaR. NEW
        * options (dict): Options for minimizing, e.g., {'maxiter': 10000}.

    Returns:
        Series {col_name: weight}
    """


    var_target_daily = var_target_annual / np.sqrt(252) if var_target_annual is not None else None

    def fitness(weights, exp_rets, covar, rf):
        mean = sum(exp_rets * weights)
        var = np.dot(np.dot(weights, covar), weights)

        if objective == 'Sharpe':
            return -(mean - rf) / np.sqrt(var)
        elif objective == 'MinRisk':
            return var
        elif objective == 'VaR_MaxReturn':
            return -mean  
        else:
            raise ValueError("Invalid objective. Choose 'Sharpe', 'MinRisk' or 'VaR_MaxReturn'.")

    def var_constraint(weights, returns, var_target, confidence_level):
        portfolio_returns = returns.dot(weights)
        calculated_var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        return var_target - calculated_var

    def asset_class_constraints(weights, asset_classes, returns, class_name, bound):
        class_weights = [weights[i] for i, asset in enumerate(returns.columns) if asset_classes[asset] == class_name]
        if bound == 'min':
            return sum(class_weights) - class_constraints[class_name]['min']
        elif bound == 'max':
            return class_constraints[class_name]['max'] - sum(class_weights)

    exp_rets = returns.mean() * 252

    if covar_method == "ledoit-wolf":
        covar = sklearn.covariance.ledoit_wolf(returns)[0] * 252
    elif covar_method == "standard":
        covar = returns.cov() * 252
    else:
        raise NotImplementedError("covar_method not implemented")

    n = len(returns.columns)
    weights = np.ones([n]) / n
    bounds = [weight_bounds for i in range(n)]

    constraints = [{'type': 'eq', 'fun': lambda W: sum(W) - 1.0}]
    for class_name in set(asset_classes.values()):
        if 'min' in class_constraints[class_name]:
            constraints.append({'type': 'ineq', 'fun': lambda W, cn=class_name: asset_class_constraints(W, asset_classes, returns, cn, 'min')})
        if 'max' in class_constraints[class_name]:
            constraints.append({'type': 'ineq', 'fun': lambda W, cn=class_name: asset_class_constraints(W, asset_classes, returns, cn, 'max')})

    if objective == 'VaR_MaxReturn' and var_target_daily is not None:
        constraints.append({'type': 'ineq', 'fun': lambda W: var_constraint(W, returns, var_target_daily, confidence_level)})

    optimized = minimize(
        fitness,
        weights,
        (exp_rets, covar, rf),
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options=options,
    )

    if not optimized.success:
        raise Exception(optimized.message)

    return pd.Series({returns.columns[i]: optimized.x[i] for i in range(n)})
w_sharpe = calc_optimal_weights_class_constraints(ret_df, asset_classes, class_constraints, objective='Sharpe')
#print(w_sharpe)
#w_min_risk = calc_optimal_weights_class_constraints(ret_df, asset_classes, class_constraints, objective='MinRisk')
#print(w_min_risk)
#w_VaR = calc_optimal_weights_class_constraints(ret_df, asset_classes, class_constraints, objective='VaR_MaxReturn', var_target_annual=-0.2, confidence_level=0.99)
#print(w_VaR)
#%%
'''
#%%
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
correlation_with_spy = pd.read_csv(os.path.join(output_dir, 'correlation.csv'), index_col=0)
corrlation_matrix = pd.read_csv(os.path.join(output_dir, 'corr.csv'), index_col=0)
corr2 = pd.read_csv(os.path.join(output_dir, 'correlation_squared.csv'),index_col=0)
betas = pd.read_csv(os.path.join(output_dir, 'Betas.csv'),index_col=0)
SPY_ret = 0.1
SPY_Stdev = 0.2
rf = 0.05
annual_ret = pd.DataFrame()
annual_ret = rf + betas*(SPY_ret-rf)
annual_var = corr2 * SPY_Stdev**2
#%%
def calc_optimal_weights_class_constraints(
    returns, 
    asset_classes, 
    class_constraints, 
    annual_ret,
    annual_var,
    correlation_matrix,
    weight_bounds=(0.01, 1.0), 
    rf=0.05, 
    covar_method="ledoit-wolf",
    objective='VaR_MaxReturn',
    hist = True,
    options={'maxiter': 10000}
):
    """
    NOTE: THIS FUNCTION IS A MODIFICATION OF 'ffn' library - calc_mean_var_weights. Please visit https://pmorissette.github.io/ffn/_modules/ffn/core.html#calc_mean_var_weights for more details. 

    Args:
        * returns (DataFrame): Returns for multiple securities.
        * asset_classes (dict): Mapping of asset to asset class. NEW
        * class_constraints (dict): Constraints for each asset class. NEW
        * weight_bounds ((low, high)): Weight limits for each asset. This is a default input in ffn, we can just use (0, 1). THIS IS NOT ASSET ALLOCATION LIMIT! 
        * rf (float): Risk-free rate used in utility calculation.
        * covar_method (str): Covariance matrix estimation method.
        * objective (str): Objective of optimal portfolio. NEW
        * var_target_annual: Annual VaR target for the portfolio. NEW
        * confidence_level: Confidence level for calculating VaR. NEW
        * options (dict): Options for minimizing, e.g., {'maxiter': 10000}.

    Returns:
        Series {col_name: weight}
    """
    def fitness(weights, exp_rets, covar, rf):
        if isinstance(exp_rets, pd.DataFrame):
            exp_rets = exp_rets.iloc[:, 0].to_numpy() 
        mean = sum(exp_rets * weights)
        var = np.dot(np.dot(weights, covar), weights)
        if objective == 'Sharpe':
            return -(mean - rf) / np.sqrt(var)
        elif objective == 'MinRisk':
            return var
        else:
            raise ValueError("Invalid objective. Choose 'Sharpe' or 'MinRisk'")
    
    def asset_class_constraints(weights, asset_classes, returns, class_name, bound):
        class_weights = [weights[i] for i, asset in enumerate(returns.columns) if asset_classes[asset] == class_name]
        if bound == 'min':
            return sum(class_weights) - class_constraints[class_name]['min']
        elif bound == 'max':
            return class_constraints[class_name]['max'] - sum(class_weights)
    
    if hist:
        exp_rets = returns.mean() * 252
        asset_names = returns.columns
        if covar_method == "ledoit-wolf":
            covar = sklearn.covariance.ledoit_wolf(returns)[0] * 252
        elif covar_method == "standard":
            covar = returns.cov() * 252
        else:
            raise NotImplementedError("covar_method not implemented")
    else:
        asset_names = annual_ret.index
        if annual_ret.empty or annual_var.empty or correlation_matrix.empty:
            raise ValueError("annual_ret, annual_var, and correlation_matrix must be provided when hist is False")
        exp_rets = annual_ret
        annual_var_array = annual_var.iloc[:, 0].to_numpy()
        covar = np.dot(np.dot(np.diag(np.sqrt(annual_var_array)), correlation_matrix), np.diag(np.sqrt(annual_var_array)))
    
    n = len(returns.columns)
    weights = np.ones([n]) / n
    bounds = [weight_bounds for i in range(n)]

    constraints = [{'type': 'eq', 'fun': lambda W: sum(W) - 1.0}]
    for class_name in set(asset_classes.values()):
        if 'min' in class_constraints[class_name]:
            constraints.append({'type': 'ineq', 'fun': lambda W, cn=class_name: asset_class_constraints(W, asset_classes, returns, cn, 'min')})
        if 'max' in class_constraints[class_name]:
            constraints.append({'type': 'ineq', 'fun': lambda W, cn=class_name: asset_class_constraints(W, asset_classes, returns, cn, 'max')})
    
    
    optimized = minimize(
        fitness,
        weights,
        (exp_rets, covar, rf),
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options=options,
    )

    if not optimized.success:
        raise Exception(optimized.message)
    optimized_weights = pd.Series({asset_names[i]: optimized.x[i] for i in range(len(asset_names))})
    
    return optimized_weights

w_sharpe = calc_optimal_weights_class_constraints(ret_df, asset_classes, class_constraints, annual_ret = annual_ret, annual_var = annual_var, correlation_matrix=corrlation_matrix, objective='Sharpe',hist=False)
print(w_sharpe)
#w_min_risk = calc_optimal_weights_class_constraints(ret_df, asset_classes, class_constraints, objective='MinRisk')
#print(w_min_risk)
#%%

#%%
port = rp.Portfolio(returns=ret_df)
method_mu='hist' 
method_cov='ledoit' 

port.assets_stats(method_mu=method_mu, method_cov=method_cov)

asset_class_list = ['USCB', 'Commodity', 'EME', 'EMB', 'USE', 'USB']

asset_classes = {'Assets': tickers,
                 'Group': asset_class_list}
asset_classes = pd.DataFrame(asset_classes)
asset_classes = asset_classes.sort_values(by=['Assets'])



constraints_dict = {
    'Disabled': [False] * len(asset_class_list),  
    'Type': ['Classes'] * len(asset_class_list),  
    'Set': ['Group'] * len(asset_class_list),     
    'Position': asset_class_list,                 
    'Sign': ['>='] * len(asset_class_list),    #Maybe add  <= 0.5 
    'Weight': [0.05] * len(asset_class_list), 
    'Type Relative': [''] * len(asset_class_list),
    'Relative Set': [''] * len(asset_class_list),
    'Relative': [''] * len(asset_class_list),
    'Factor': [''] * len(asset_class_list)
}

# Create the DataFrame
constraints = pd.DataFrame(constraints_dict)

A, B = rp.assets_constraints(constraints, asset_classes)



port.ainequality = A
port.binequality = B

model = 'Classic'
rm = 'MV'
obj = 'Sharpe' # ' MinRisk'
l=0
rf = 0.001

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=True)

print(w)
'''
