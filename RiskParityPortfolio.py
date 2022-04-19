import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import riskfolio as rp

""""
Metholodogy
- Sliding window approach
- Run Risk parity model on ith year
- Obtain weights from ith year and use on (i+1)th year
- Simultaneously calculate both portfolio performance using Volatility, Returns, Sharpe Ratio for baseline(1/3 weight for each asset) and Risk Parity Weights on (i+1)th year
- Compare metrics in pandas dataframe

Assume
- risk free rate of 2.85% used in sharpe ratio
- assume no transaction cost/commission rate
"""



# Run test on y1, test performance on y2
# Download
def download_data(cur_year, returns = 0):
    # start and end of year
    first_day_of_year = datetime.date.min.replace(year = cur_year)
    last_day_of_year = datetime.date.max.replace(year = cur_year)

    # download for assets
    assets = ['SPY', 'GLD', 'TLT']
    assets.sort()
    data = yf.download(assets, start = first_day_of_year, end = last_day_of_year)
    data = data.loc[:,('Adj Close', slice(None))]
    data.columns = assets

    if returns:
        return data, data[assets].pct_change().dropna()
    return data

# Model
def model_rpp(returns_daily):
    # Building the portfolio object
    port = rp.Portfolio(returns = returns_daily)

    # Calculating optimal portfolio

    # Select method and estimate input parameters:

    method_mu='hist' # Method to estimate expected returns based on historical data.
    method_cov='hist' # Method to estimate covariance matrix based on historical data.

    port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

    # Estimate optimal portfolio:

    model='Classic' # use estimates of expected return vector and covariance matrix that depends on historical data
    rm = 'MV' # Risk measure used, Standard Deviation
    hist = True # Use historical scenarios for risk measures that depend on scenarios
    rf = 0 # Risk free rate (the rate of return of an investment with no risk of loss)
    b = None # vector of risk constraints per asset 
    w_rp = port.rp_optimization(model=model, rm=rm, rf=rf, b=b, hist=hist)
    return w_rp

# obtain weights for specific asset
def obtain_weight(rp_df, asset):
    #assets = ['SPY', 'GLD', 'TLT']
    return rp_df.loc[asset, "weights"]

# Test on y2  
# start price in y2
def obtain_start_price(asset, df):
    return df.iloc[0, :][asset] 

# final price end of y2 for specific asset
def obtain_end_price(asset, df):
    return df.iloc[-1, :][asset]

def calculate_volatility(returns, alloc_pct):
    covariance = np.cov(returns.T)
    if type(alloc_pct) == type(pd.DataFrame()):
        weight_vector = alloc_pct.to_numpy().T
    else:
        weight_vector = alloc_pct
        
    portfolio_volatility = np.sqrt(np.dot(np.dot(weight_vector, covariance), weight_vector.T))
    return portfolio_volatility[0][0]

def calculate_sharpe_ratio(returns):
    # risk free rate from US treasury bond is 2.85%, 252 trading days annualized
    num_trading_days = 252
    sqrt_day = np.sqrt(num_trading_days)
    return (returns.mean()*num_trading_days - 0.0285)/(returns.std()*np.sqrt(num_trading_days))


start_year = 2010
end_year = 2019

def back_test(start_year, end_year):
    portfolio_volatility = []
    baseline_portfolio_volatility = []
    annual_returns = []
    baseline_annual_returns = []
    sharpe_list = []
    baseline_sharpe_list = []

    base = 0
    init = 0

    downloaded_data = [False for i in range(start_year, end_year)]
    
    # Year Test
    for cur_year in range(start_year, end_year):
        if cur_year < end_year-1:
            assets = ['SPY', 'GLD', 'TLT']
            assets.sort()

            # to set index for downloaded_data
            if init == 0:
                base_year = cur_year
                init = 1

            # Store to avoid repeated downloads
            # Download data, Y1
            idx_i = cur_year-base_year
            # not present, download
            if not downloaded_data[idx_i]:
                downloaded_data_y1, downloaded_data_y1_returns = download_data(cur_year, True)
                downloaded_data[idx_i] = [downloaded_data_y1, downloaded_data_y1_returns]
            # if present, retrieve
            else:
                downloaded_data_y1, downloaded_data_y1_returns = downloaded_data[idx_i][0], downloaded_data[idx_i][1]

            if not downloaded_data[idx_i+1]:
                downloaded_data_y2, downloaded_data_y2_returns = download_data(cur_year + 1, True)
                downloaded_data[idx_i+1] = [downloaded_data_y2, downloaded_data_y2_returns]
            else:
                downloaded_data_y2, downloaded_data_y2_returns = downloaded_data[idx_i+1][0], downloaded_data[idx_i+1][1]

            # deep copy for separate baseline values
            baseline_downloaded_data_y2 = downloaded_data_y2.copy(deep = True)

            # starting variables
            alloc_pct = model_rpp(downloaded_data_y1_returns)
            returns_df = {}
            starting_amt = 100000
            new_portfolio = 0
            new_baseline_portfolio = 0

            for asset in assets:
                weight =  obtain_weight(alloc_pct, asset)
                # RPP
                budget = weight * starting_amt
                # Baseline
                baseline_budget = (1/3) * starting_amt

                # start price in y2
                start_price = obtain_start_price(asset, downloaded_data_y2)
                # RPP
                position_size = budget/start_price
                downloaded_data_y2[asset] = downloaded_data_y2[asset] * position_size

                # Baseline
                baseline_position_size  = baseline_budget/start_price
                baseline_downloaded_data_y2[asset] = baseline_downloaded_data_y2[asset] * baseline_position_size


            downloaded_data_y2["Total"] = downloaded_data_y2[list(downloaded_data_y2.columns)].sum(axis = 1)
            baseline_downloaded_data_y2["Total"] = baseline_downloaded_data_y2[list(baseline_downloaded_data_y2.columns)].sum(axis = 1)
            
            new_portfolio = obtain_end_price("Total", downloaded_data_y2)
            new_baseline_portfolio = obtain_end_price("Total", baseline_downloaded_data_y2)

            # calculate sharpe ratio
            sharpe = calculate_sharpe_ratio(downloaded_data_y2["Total"].pct_change(1).dropna())
            baseline_sharpe = calculate_sharpe_ratio(baseline_downloaded_data_y2["Total"].pct_change(1).dropna())
            sharpe_list.append(sharpe)
            baseline_sharpe_list.append(baseline_sharpe)

            # volatility
            volatility = calculate_volatility(downloaded_data_y2_returns, alloc_pct)
            baseline_volatility = calculate_volatility(downloaded_data_y2_returns, np.array([[1/3]*3]))

            # baseline volatility
            portfolio_volatility.append(volatility)
            baseline_portfolio_volatility.append(baseline_volatility)

            # overall returns 
            overall_portfolio_returns = ((new_portfolio-starting_amt)/starting_amt) * 100
            baseline_overall_portfolio_returns = ((new_baseline_portfolio-starting_amt)/starting_amt) * 100

            # baseline overall returns
            annual_returns.append(overall_portfolio_returns)
            baseline_annual_returns.append(baseline_overall_portfolio_returns)

    eval_df = pd.DataFrame({"portfolio_volatility" : portfolio_volatility, "baseline_portfolio_volatility" :baseline_portfolio_volatility,
                 "annual_returns" : annual_returns, "baseline_annual_returns":annual_returns,
                  "sharpe_ratio":sharpe_list, "baseline_sharpe_ratio":baseline_sharpe_list}, index = [i for i in range(start_year+1, end_year)])

    # to show in command prompt, must print
    print(eval_df)
    return eval_df

if __name__ == "__main__":
    back_test(start_year, end_year)
