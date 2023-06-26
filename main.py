import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
# import streamlit as st


class Portfolio:
    def __init__(self, symbols):
        self.symbols = symbols
        self.data = self._get_data()
        self.mean = self.data.mean()
        self.variance = self.data.var()
        self.covariance = self.data.cov()

    # @st.cache
    def _get_data(self):
        # Get data from Yahoo Finance API
        data = yf.download(self.symbols, start='2019-03-10', end='2021-03-10', group_by='ticker')['Adj Close']

        return data
    
    def simulate(self, n_portfolios, n_years=2, rf_rate=0.01):
        # Monte Carlo simulation of n_portfolios random portfolios
        np.random.seed(42)
        returns = np.log(self.data/self.data.shift(1)).dropna()
        mean_return = returns.mean()
        cov_matrix = returns.cov()
        num_assets = len(self.symbols)
        portfolio_returns = []
        portfolio_volatility = []
        portfolio_weights = []
        for portfolio in range(n_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            returns_annual = np.sum(mean_return * weights) * n_years
            volatility_annual = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * n_years, weights)))
            portfolio_returns.append(returns_annual)
            portfolio_volatility.append(volatility_annual)
            portfolio_weights.append(weights)
        portfolio_returns = np.array(portfolio_returns)
        portfolio_volatility = np.array(portfolio_volatility)
        portfolio_weights = np.array(portfolio_weights)
        sharpe_ratios = (portfolio_returns - rf_rate) / portfolio_volatility
        max_sharpe_idx = sharpe_ratios.argmax()
        optimal_weights = portfolio_weights[max_sharpe_idx]
        return portfolio_returns, portfolio_volatility, portfolio_weights, optimal_weights
    
    def plot_correlation_matrix(self):
        # Plot correlation matrix using seaborn
        plt.figure(figsize=(10,8))
        sns.heatmap(self.covariance, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()
    

    def plot_efficient_frontier(self, portfolio_returns, portfolio_volatility, optimal_weights, rf_rate):
        frontier_returns, frontier_volatility = self.calculate_efficient_frontier(portfolio_returns, rf_rate)
        sharpe_ratios = (frontier_returns - rf_rate) / frontier_volatility
        sharpe_idx = np.argmax(sharpe_ratios)

        # Plot efficient frontier using Matplotlib
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(portfolio_volatility, portfolio_returns, c=(portfolio_returns / portfolio_volatility), marker='o')
        ax.scatter(portfolio_volatility[optimal_weights == 1], portfolio_returns[optimal_weights == 1], c='red', marker='*', s=300)
        ax.scatter(frontier_volatility, frontier_returns, c=sharpe_ratios, cmap='viridis', marker='.')
        ax.scatter(frontier_volatility[sharpe_idx], frontier_returns[sharpe_idx], c='red', marker='*', s=300)
        ax.plot(frontier_volatility, frontier_returns, 'g--', linewidth=2)
        ax.set_title('Efficient Frontier')
        ax.set_xlabel('Volatility (Std. Deviation)')
        ax.set_ylabel('Expected Returns')
        fig.colorbar(ax.scatter(portfolio_volatility, portfolio_returns, c=(portfolio_returns / portfolio_volatility)), label='Sharpe Ratio')
        plt.show(fig) 

    def save_results(self, portfolio_returns, portfolio_volatility, optimal_weights, filename):
        # Save results to file
        plt.figure(figsize=(10,8))
        sns.heatmap(self.covariance, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show() 
        plt.close()
        plt.figure(figsize=(10,8))
        plt.scatter(portfolio_volatility, portfolio_returns, c=(portfolio_returns / portfolio_volatility))
        plt.show()
