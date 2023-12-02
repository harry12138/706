import pandas as pd
import yfinance as yf
import os

os.chdir(os.path.dirname(__file__))
data_dir = os.path.join(os.getcwd(), "Data")
output_dir = os.path.join(os.getcwd(), "Output")

#Download Data from Yahoo Finance
# ETF_li = ['SPY', 'QQQ', 'DIA', 'IWF', 'EEM', 'MCHI', 'INDA', 'TLT', 'IEF', 'VGSH', 'LQD', 'HYG', 'DBC', 'EMB']
#ETF_li = ['SPY',  'EEM', 'TLT', 'AGG', 'DBC', 'EMB', 'HYG'] #'EZU','EWC',
ETF_li = ['QQQ', 'DIA', 'INDA', 'MCHI', 'TLT', 'LQD', 'HYG', 'DBC', 'EMB', 'VGSH']
# What if we use ['QQQ', 'DIA', 'INDA', 'MCHI', 'TLT', 'LQD', 'HYG', 'DBC', 'EMB', 'VGSH']
start_date = '2013-11-01'
end_date = '2023-10-31'

if not os.path.exists(os.path.join(data_dir, 'stock_data')):
    os.makedirs(os.path.join(data_dir, 'stock_data'))

for etf in ETF_li:
    data = yf.download(etf, start=start_date, end=end_date)
    data.to_csv(os.path.join(data_dir, 'stock_data', f'{etf}.csv'))

