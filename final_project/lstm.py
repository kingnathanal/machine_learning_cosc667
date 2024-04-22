##################
#
# William Britton
#
# This script demonstrates how to use LSTM to predict stock prices.
# It uses the yfinance library to download stock price data, preprocesses the data, and trains an LSTM model.
# The script then evaluates the model using the mean squared error and plots the predictions against the actual data.
# LSTM (Long Short-Term Memory) is a type of recurrent neural network that is well-suited for time series data.
# It can learn long-term dependencies and has been successfully used in various applications, including stock price prediction.
#
##################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

# Fetch stock data from Yahoo Finance
def fetch_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Close'].values.reshape(-1, 1)

# Scale and split the data
def preprocess_data(stock_data, sequence_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data_scaled = scaler.fit_transform(stock_data)

    X, y = [], []

    for i in range(sequence_length, len(stock_data_scaled)):
        X.append(stock_data_scaled[i-sequence_length:i])
        y.append(stock_data_scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    return X, y, scaler

# Build the LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main execution function
def main():
    ticker = 'RBLX'  # Example stock ticker, change as needed
    start_date = '2010-01-01'
    end_date = '2023-01-01'

    #stock_data = fetch_data(ticker, start_date, end_date)
    stock = yf.download(ticker, start=start_date, end=end_date)
    #actual_prices = stock['Close'].values
    stock_data = stock['Close'].values.reshape(-1, 1)
    sequence_length = 60  # Number of days to look back for prediction

    X_train, y_train, scaler = preprocess_data(stock_data, sequence_length)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=25, batch_size=32)

    ## Predict the stock price with real data

    test_stock = yf.download(ticker, start='2023-01-01', end='2024-04-20')
    actual_prices = test_stock['Close'].values

    total_dataset = pd.concat((stock['Close'], test_stock['Close']), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_stock) - sequence_length:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []

    for i in range(sequence_length, len(model_inputs)):
        x_test.append(model_inputs[i-sequence_length:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predict and evaluate the model
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)  # Unscale the predictions

    results_df = pd.DataFrame({
        'Actual Price': actual_prices.flatten(),
        'Predicted Price': predictions.flatten()
    })
    print(results_df.tail(20))


    # Plot the results
    plt.figure(figsize=(12, 8))
    #plt.plot(y_test_unscaled, label='True Price')
    plt.plot(actual_prices, label='True Price')
    plt.plot(predictions, label='Predicted Price')
    plt.title('Stock Price Prediction - ' + ticker)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    real_data = [model_inputs[len(model_inputs) + 1 - sequence_length:len(model_inputs+1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f'Prediction: {prediction}')


if __name__ == '__main__':
    main()
