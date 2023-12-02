import pandas as pd
import os
import numpy as np
import yfinance as yf
import statsmodels.api as sm

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

start_date = '2013-11-01'
end_date = '2023-10-31'
SPY = yf.download('SPY', start=start_date, end=end_date)
SPY_ret = SPY[['Adj Close']].pct_change().dropna()
SPY_ret.rename(columns={'Adj Close':'SPY'}, inplace = True)

ret_df.index = pd.to_datetime(ret_df.index).normalize()
SPY_ret.index = pd.to_datetime(SPY_ret.index).normalize()

aligned_data = ret_df.join(SPY_ret, how='inner', rsuffix='_SPY')
SPY_aligned_ret = aligned_data['SPY']
Corr = aligned_data.corr().iloc[:-1, -1]
corr_squared = Corr ** 2

ret_aligned_df = aligned_data.drop('SPY', axis=1)

def calculate_betas(returns, market_returns):
    betas = {}
    for asset in returns.columns:
        Y = returns[asset]
        X = market_returns
        X = sm.add_constant(X) 
        model = sm.OLS(Y, X).fit()
        betas[asset] = model.params[1]  # Beta coefficient
    return pd.Series(betas)



betas = calculate_betas(ret_aligned_df, SPY_aligned_ret)
df = pd.DataFrame(betas)
df.to_csv(os.path.join(output_dir, 'Betas.csv'))
corr_squared.to_csv(os.path.join(output_dir, 'correlation_squared.csv'))
Corr.to_csv(os.path.join(output_dir, 'correlation.csv'))
