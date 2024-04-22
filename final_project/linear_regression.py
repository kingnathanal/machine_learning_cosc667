####################
# 
# William Britton
#
# This script demonstrates how to use linear regression to predict stock prices.
# It uses the yfinance library to download stock price data, and the scikit-learn library to build and evaluate the model.
# The script uses the Close price as the target variable and the days as the feature for simplicity.
# The script then evaluates the model using the mean squared error and R^2 score.
# Finally, the script plots the predictions against the actual data.
#
####################

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Load stock price data
ticker = 'AAPL'  # Example: Apple Inc.
start_date = '2023-11-01'
end_date = '2024-04-16'
data = yf.download(ticker, start=start_date, end=end_date)

# Use Close price as the target variable
X = np.array(range(len(data))).reshape(-1, 1)  # Use days as feature for simplicity
y = data['Close'].values

# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create models
models = {
    "Linear Regression": LinearRegression(),
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: MSE = {mse:.2f}, R^2 = {r2:.2f}")
    # Plot predictions
    plt.figure(figsize=(12, 8))
    plt.scatter(X_test, y_test, color='black', label='Actual data')
    plt.plot(X_test, y_pred, label=f'Predicted by {name}', linewidth=2)
    plt.title(f'{ticker} - {name} Prediction vs Actual')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()