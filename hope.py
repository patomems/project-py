import pandas
from pandas_datareader import data as pdr
# import pandas_datareader as pdr

import yfinance as yfin


yfin.pdr_override()

assets = ['AAPL', 'GOOG', 'AMZN', 'NFLX', 'TSLA', 'NVDA']

start_date='2022-10-24', 
end_date='2022-12-23'
spy=pdr.get_data_yahoo(assets, start='2022-10-24', end='2022-12-23')

# spy = pdr.get_data_yahoo('SPY', start='2022-10-24', end='2022-12-23')

print(spy)
