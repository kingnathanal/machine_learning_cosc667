#####################
# Polynomial Regression
#
# William Britton
#
# This script demonstrates how to use polynomial regression to predict stock prices.
# It uses the yfinance library to download stock price data, and the scikit-learn library to build and evaluate the model.
# The script uses the Close price as the target variable and the days as the feature for simplicity.
# The script uses the PolynomialFeatures class to generate polynomial features, and the LinearRegression class to build the model.
# The script then evaluates the model using the mean squared error and R^2 score.
# Finally, the script plots the predictions against the actual data.
#
#####################

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load stock price data
ticker = 'AAPL'  # Example: Apple Inc.
start_date = '2024-01-01'
end_date = '2024-03-11'
data = yf.download(ticker, start=start_date, end=end_date)

print(data.keys())

df = pd.DataFrame(data)
df.sort_values('Date', inplace=True)
df.reset_index(inplace=True)

df['Date'] = df['Date'].apply(lambda x: x.toordinal())
# Use Close price as the target variable
X = df['Date'].values.reshape(-1, 1)  # Use days as feature for simplicity
y = df['Close'].values.reshape(-1, 1)

poly_features = PolynomialFeatures(degree=2)  # Using 2nd degree polynomial features
X_poly = poly_features.fit_transform(X)

# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Polynomial Regression: MSE = {mse:.2f}, R^2 = {r2:.2f}")

# Plot predictions
plt.scatter(X_test[:, 1], y_test, color='black', label='Actual data')
plt.plot(X_test[:, 1], y_pred, label='Predicted by Polynomial Regression', linewidth=2)
plt.title('Polynomial Regression Prediction vs Actual')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()