#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 17:25:30 2024
@author: Tiffany
"""

# Analyze historical sales data for office supplies.
# Model the time series using the Seasonal ARIMA (SARIMA) framework.
# Forecast future sales and evaluate the model's accuracy.

# load libraries	
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore") # Suppress warnings for a cleaner output
plt.style.use('fivethirtyeight') # Set plot style for better visualization
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib
from pylab import rcParams

# Configure matplotlib parameters for consistent and readable plots
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

# Load the dataset from a local Excel file and focus on the 'Office Supplies' category
df = pd.read_excel("/Users/XXX/Superstore.xls")
supplies = df.loc[df['Category'] == 'Office Supplies']

# Clean and preprocess the data:
# - Remove unnecessary columns
# - Sort by 'Order Date' to prepare for time series analysis
cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 
        'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 
        'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']

supplies.drop(cols, axis=1, inplace=True)
supplies = supplies.sort_values('Order Date')
supplies.isnull().sum()

# Set the 'Order Date' column as the index and ensure it's formatted as a time series
supplies = supplies.set_index('Order Date')
supplies.index

# Resample the sales data to calculate the mean sales for each month
y = supplies['Sales'].resample('MS').mean()

# Check for stationarity in the time series using the Augmented Dickey-Fuller (ADF) test
result = adfuller(y)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# Visualize the time series data up to 2017
y['2017':].plot(figsize=(15, 6))
plt.show()

# Decompose the time series into trend, seasonal, and residual components
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
decomposition.plot()
plt.show()

# Perform a grid search to find the best parameters for a Seasonal ARIMA (SARIMA) model
p = d = q = range(0, 2)  # Define ranges for the model parameters
pdq = list(itertools.product(p, d, q))  # Generate combinations of ARIMA parameters
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]  # Add seasonal parameters

print('Examples of parameter combinations for Seasonal ARIMA...')
for param in pdq[:3]:  # Display a few examples
    for param_seasonal in seasonal_pdq[:2]:
        print(f'SARIMAX: {param} x {param_seasonal}')

# Fit various SARIMA models and compare them using the AIC score
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print(f'ARIMA{param}x{param_seasonal} - AIC: {results.aic}')
        except:
            continue

# Select the best model based on AIC and fit it
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 0),
                                seasonal_order=(1, 1, 0, 12))
results = mod.fit()
print(results.summary().tables[1])

# Generate forecasts and compare them with actual values
pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Office Supply Sales')
plt.legend()
plt.show()

# Calculate the forecast error (Mean Squared Error and Root Mean Squared Error)
y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print(f'The Mean Squared Error of our forecasts is {round(mse, 2)}')
print(f'The Root Mean Squared Error of our forecasts is {round(np.sqrt(mse), 2)}')

# Produce a long-term forecast and visualize it
pred_uc = results.get_forecast(steps=12)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()
