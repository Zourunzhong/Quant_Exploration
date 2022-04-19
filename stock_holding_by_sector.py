import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import riskfolio as rp

""""
using yfinance , iterate through all stock holdings in each fund to add up weighted counts of each sector

- info is very slow
- .info information is stored within each run for fast retrieval subsequently to minimise its call



"""

def main():
    fund_ticker_list = {"CSPX": 0.5, "EFA": 0.3, "QQQ": 0.2}
    sector_holdings = {}
    stock_ticker_dict = {}
    all_fund_stock_ticker_list = {}

    for fund_ticker in fund_ticker_list:
        # store for future reference to info object
        fund_info = yf.Ticker(fund_ticker)
        holdings_dict_list = fund_info.info['holdings']
        fund_stock_ticker_list = list(map(lambda x: x['symbol'], holdings_dict_list))
        # store so outer loop does not need to be rerun if inner loop fails
        all_fund_stock_ticker_list[fund_ticker] = fund_stock_ticker_list

    for fund_ticker in fund_ticker_list:
        # obtain tickers of all stocks in the fund
        fund_stock_ticker_list = all_fund_stock_ticker_list[fund_ticker]

        # iterate through each ticker
        for stock_ticker in fund_stock_ticker_list:

            # if sector not already obtained
            if stock_ticker not in stock_ticker_dict:
                # obtain from .info - slow
                stock_info = yf.Ticker(stock_ticker).info

                # stock ticker not in stock_ticker_dict and stock_sector is in stock_info
                # retrieve stock_sector from stock_info
                if "sector" in stock_info:
                    stock_sector = stock_info['sector']

                else:
                    # edge case
                    if stock_ticker == "BRK.B":
                        stock_sector = "Financial Services"


                # store sector into stock_ticker_dict for future reference
                stock_ticker_dict[stock_ticker] = stock_sector
            else:
                stock_sector = stock_ticker_dict[stock_ticker]

            # portfolio allocation weight
            weight_of_fund = fund_ticker_list[fund_ticker]

            # add to count for each sector
            if stock_sector not in sector_holdings:
                sector_holdings[stock_sector] = weight_of_fund

            else:
                sector_holdings[stock_sector] += weight_of_fund

    sector_holdings_df = pd.DataFrame(sector_holdings, index=[0, ])
    total = sector_holdings_df.sum(axis=1)[0]

    print(sector_holdings_df / total)
    return sector_holdings_df / total


if __name__ == "__main__":
    main()