# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime as dt
# import pytz
# import seaborn as sns

# import scipy.optimize as sco
# from pandas_datareader import data as pdr

# import yfinance as yf
# yf.pdr_override()

# class Portfolio:


#     def __init__(self, assets):
#         self.assets = assets
#         self.mean_returns = None
#         self.cov_matrix = None
#         self.num_assets = len(assets)
#         self.weights = np.random.random(self.num_assets)
#         self.weights /= np.sum(self.weights)


#     def download_data(self, start_date, end_date):
#         yf.pdr_override()

#         data=pdr.get_data_yahoo(self.assets, start=start_date, end=end_date)

#         print(data)
#         self.daily_returns = data['Adj Close'].pct_change()
#         self.mean_returns = self.daily_returns.mean()

#         print(self.mean_returns)

#         self.cov_matrix = self.daily_returns.cov().to_numpy()


#     def calculate_portfolio_performance(self, weights):
#         portfolio_return = np.dot(self.mean_returns, weights)
#         portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
#         sharpe_ratio = portfolio_return / portfolio_volatility

#         return -sharpe_ratio


#     def generate_efficient_frontier(self, num_portfolios):
#         results = np.zeros((3, num_portfolios))
#         for i in range(num_portfolios):
#             weights = np.random.random(self.num_assets)
#             weights /= np.sum(weights)
#             portfolio_return = np.dot(self.mean_returns, weights)
#             portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
#             sharpe_ratio = portfolio_return / portfolio_volatility
#             results[0, i] = portfolio_return
#             results[1, i] = portfolio_volatility
#             results[2, i] = sharpe_ratio
#         return results


#     def optimize_portfolio(self):
#         constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
#         bounds = tuple((0, 1) for i in range(self.num_assets))
#         initial_weights = self.num_assets * [1. / self.num_assets]
#         optimal_weights = sco.minimize(self.calculate_portfolio_performance, initial_weights, method='SLSQP',
#                                        bounds=bounds, constraints=constraints)
#         return optimal_weights.x


# # yf.pdr_override()

# # Set up the assets
# assets = ['AAPL', 'GOOG', 'AMZN',  'NFLX', 'TSLA', 'NVDA','SPY']
# portfolio = Portfolio(assets)

# # # Download the historical data

# start_date='2020-10-24', 
# end_date='2022-10-24'
# # start_date = '2019-03-14'
# # end_date = '2021-03-14'
# portfolio.download_data( '2020-10-24', '2022-10-24')

# # Calculate the mean, variance, and correlation matrix for all assets
# mean_returns = portfolio.mean_returns
# cov_matrix = portfolio.cov_matrix
# corr_matrix = portfolio.daily_returns.corr()

# # Calculate the efficient frontier and optimal weights for the portfolio
# num_portfolios = 5000
# results = portfolio.generate_efficient_frontier(num_portfolios)
# optimal_weights = portfolio.optimize_portfolio()

# # Graph the results
# sns.set()
# plt.figure(figsize=(10, 7))
# plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis')
# plt.colorbar(label='Sharpe Ratio')
# plt.xlabel('Volatility')
# plt.ylabel('Return')
# plt.scatter(results[1, :].min(), results[0, :].max(), marker='*', s=500, c='r', label='Optimal Portfolio')
# plt.legend()
# plt.title('Efficient Frontier')
# plt.show()

# print('Optimal weights:', optimal_weights)




1. Import dataset "ufo_sightings_large.csv" in pandas

import numpy as np
import pandas as pd
df = pd.read_csv('ufo_sightings_large.csv')
print(df.country)

2. Checking column types & Converting Column types 
Take a look at the UFO dataset's column types using the dtypes attribute. Please convert the column types to the proper types.
For example, the date column, which can be transformed into the datetime type. 
That will make our feature engineering efforts easier later on.

#pd.options.mode.chained_assignment = None
df['date'] = pd.to_datetime(df['date'])
df['type'] = df['type'].astype('category')
print(df.length_of_time)


3. Dropping missing data 
Let's remove some of the rows where certain columns have missing values. 

# Drop rows with missing values
df = df.dropna()

 4. Extracting numbers from strings 
The <b>length_of_time</b> column in the UFO dataset is a text field that has the number of 
minutes within the string. 
Here, you'll extract that number from that text field using regular expressions.

# Extract numbers from 'length_of_time' column using regular expressions and create a new column
df['duration_minutes'] = df['length_of_time'].str.extract('(\d+)').astype(float)

# Convert duration_minutes to minutes if 'hour' or 'min' is in length_of_time column
df.loc[df['length_of_time'].str.contains('hour'), 'duration_minutes'] *= 60
df.loc[df['length_of_time'].str.contains('min'), 'duration_minutes']
df.loc[df['length_of_time'].str.contains('seconds'), 'duration_minutes'] /= 60

print(df.country)



5. Identifying features for standardization 
In this section, you'll investigate the variance of columns in the UFO dataset to 
determine which features should be standardized. You can log normlize the high variance column.

# Identify columns to standardize
features_to_standardize = ['length_of_time','type', 'country']

6. Encoding categorical variables 
There are couple of columns in the UFO dataset that need to be encoded before they can be 
modeled through scikit-learn. 
You'll do that transformation here, <b>using both binary and one-hot encoding methods
This link can help; https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding/ </b>.

# !pip install category_encoders


# Encode categorical variables using one-hot encoding
# !pip install category_encoders
import category_encoders as ce
# x=df.iloc[:,:].values
from sklearn.preprocessing import LabelEncoder
# labelEncoder_x=LabelEncoder()
# # x[:,1]=labelEncoder_x.fit_transform(x[:,1])
# x[:,1]=labelEncoder_x.fit_transform(x[:,1])
# y=pd.DataFrame(x)
cols=['city','state','country','type']
df[cols]=df[cols].apply(LabelEncoder().fit_transform)
df.head(10)


7. Text vectorization
Let's transform the desc column in the UFO dataset into tf/idf vectors, 
since there's likely something we can learn from this field.

#from sklearn.feature_extraction.text import CountVectorizer
#vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
#desc_vectors = vectorizer.fit_transform(ufo_sightings['desc'])

# Create a DataFrame from the vectorized text
#desc_df = pd.DataFrame(desc_vectors.toarray(), columns=vectorizer.get_feature_names())

# Concatenate the vectorized text DataFrame with the original DataFrame
#df = pd.concat([u

8. Selecting the ideal dataset
Let's get rid of some of the unnecessary features. 

# Select relevant columns for analysis
import numpy as np


## 9. Split the X and y using train_test_split, setting stratify = y (5 points)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

10. Train (fit) ML model KNeighborsClassifier (knn with n_neighbors=5) using Scikit Learn to the training sets, and print the score of knn on the test sets

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
# Fit knn to the training sets
knn.fit(X_train, y_train)
# Print the score of knn on the test sets
score = knn.score(X_test, y_test)
print(knn.score(X_test, y_test))


