# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 17:03:28 2023

@author: harry
"""

import pandas as pd
import os
import riskfolio as rp
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

corrleation = ret_df.corr()
print(corrleation)
corrleation.to_csv(os.path.join(output_dir, 'corr.csv'))