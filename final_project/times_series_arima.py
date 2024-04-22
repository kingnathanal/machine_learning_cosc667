#####################
# Time Series Analysis with ARIMA
#
# William Britton
# 
# This script demonstrates how to use the ARIMA model to forecast stock prices.
# It uses the yfinance library to download historical stock price data and the statsmodels library to build the ARIMA model.
# The script fits the ARIMA model to the historical data and forecasts the next 5 days.
# Finally, the script prints the forecasted values.
#
#####################

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Fetch historical data for AAPL
aapl = yf.download('RBLX', start='2024-01-01', end='2024-04-14')

# Keep only the 'Close' column
aapl_close = aapl[['Close']]

# Display the head of the DataFrame
print(aapl_close.head())

# Define the model
model = ARIMA(aapl_close, order=(5,1,0))  # Example parameters; adjust based on your data analysis

# Fit the model
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# Forecast the next 5 days
forecast = model_fit.forecast(steps=5)

print(forecast)