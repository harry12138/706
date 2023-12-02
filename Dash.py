import pandas as pd
import os
os.chdir(os.path.dirname(__file__))
#%%
import numpy as np
from PortfolioOpt import calc_optimal_weights_class_constraints
import sklearn.covariance
import sklearn.cluster
# import matplotlib.pyplot as plt
# import seaborn as sns
from Monte_Carlo import user_input, monte_carlo_simulation
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
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px

# Create the Dash application
app = dash.Dash(__name__)
server = app.server

# Application layout
app.layout = html.Div([
    dcc.Dropdown(
        id='portfolio_type',
        options=[
            {'label': 'Sharpe', 'value': 'Sharpe'},
            {'label': 'MinRisk', 'value': 'MinRisk'},
            {'label': 'VaR_MaxReturn', 'value': 'VaR_MaxReturn'}
        ],
        value='Sharpe',
        placeholder="Select Portfolio Type"
    ),
    html.Br(),
    dcc.Input(id='var_target_annual', type='number', placeholder='Enter VAR Target Annual'),
    html.Br(),
    dcc.Input(id='confidence_level', type='number', placeholder='Enter Confidence Level'),
    html.Br(),
    html.Div([
        html.Label('Use Default Setting (Historical Data)'),
        dcc.Checklist(
            id='default_setting',
            options=[
                {'label': '', 'value': 'default'}
            ],
            value=['default']
        )
    ]),
    html.Br(),
    dcc.Input(id='SPY_annual_ret', type='number', placeholder='Enter SPY Annual Return'),
    html.Br(),
    dcc.Input(id='SPY_annual_stdev', type='number', placeholder='Enter SPY Annual Std Dev'),
    html.Br(),
    dcc.Input(id='num_periods', type='number', placeholder='Enter Number of Periods', value=252),
    html.Br(),
    html.Button('Run Simulation', id='submit_button', n_clicks=0),
    html.Div(id='simulation_output'),
    dcc.Graph(id='histogram_output')  # Graph component for displaying histogram
])

# Callback for running the Monte Carlo simulation and displaying histogram
@app.callback(
    [Output('simulation_output', 'children'),
     Output('histogram_output', 'figure')],
    Input('submit_button', 'n_clicks'),
    State('portfolio_type', 'value'),
    State('var_target_annual', 'value'),
    State('confidence_level', 'value'),
    [State('default_setting', 'value')],
    State('SPY_annual_ret', 'value'),
    State('SPY_annual_stdev', 'value'),
    State('num_periods', 'value')
)
def update_output(n_clicks, type_, var_target_annual, confidence_level, default_setting, SPY_annual_ret, SPY_annual_stdev, num_periods):
    if n_clicks > 0:
        use_default = 'default' in default_setting

        # Call the monte_carlo_simulation function with user inputs
        simulation_result = monte_carlo_simulation(
            type_, var_target_annual, confidence_level, SPY_annual_ret, SPY_annual_stdev, 
            num_periods, num_simulations=10000, hist=use_default
        )
        total_returns = simulation_result.iloc[:, -1]

        # Create histogram
        fig = px.histogram(
            total_returns, 
            nbins=50, 
            title='Histogram of Total Returns Across All Simulations'
        )
        fig.update_layout(xaxis_title='Total Return', yaxis_title='Frequency')

        return f'Simulation Completed. Result: {simulation_result}', fig
    return 'Enter values and run the simulation.', {}

# Run the application
if __name__ == '__main__':
    app.run_server(debug=False)
