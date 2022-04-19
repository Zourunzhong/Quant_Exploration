# Quant_Exploration



https://www.youtube.com/watch?v=Lot30tsEhy0

3 
naive risk parity
	- inverse volatyility weighting
		- most volatile lowest weight
equal risk contribution
	- equal risk contribution method
		- historical correlations of assets/ strategies
			
maximum diversification
	- maxiumum diversification optimization substitutes asset 
		volatilities for returns in max sharpe ratio optim


equally weighted benchmark (baseline)

Implementation
https://medium.com/@orenji.eirl/vanilla-risk-parity-with-python-and-riskfolio-lib-3dfbfb752067

Risk free Rate
https://ycharts.com/indicators/10_year_treasury_rate#:~:text=10%20Year%20Treasury%20Rate%20is%20at%202.85%25%2C%20compared%20to%202.83,day%20and%201.59%25%20last%20year.

Volatility:
https://whynance.medium.com/use-python-to-estimate-your-portfolios-volatility-eee22d1a37db

Sharpe Ratio:
https://www.codearmo.com/blog/sharpe-sortino-and-calmar-ratios-python
https://blog.quantinsti.com/sharpe-ratio-applications-algorithmic-trading/

Constructing dataframe:
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html

Numpy element wise multiplication/division (Hadamard):
np.divide()

First/Last Date of the year:
import datetime
year = 2016
first_day_of_year = datetime.date.min.replace(year = year)
last_day_of_year = datetime.date.max.replace(year = year)
print(first_day_of_year, last_day_of_year)
