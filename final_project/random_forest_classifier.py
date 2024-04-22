####################################################################################################
#
# William Britton
#
# Description: This script uses a Random Forest Classifier to predict stock price movements.
# It downloads stock price data using the yfinance library, preprocesses the data, and trains a Random Forest Classifier model.
# The script then evaluates the model using the precision score and plots the predictions against the actual data.
# The Random Forest Classifier is a popular machine learning algorithm that can be used for classification tasks.
# It works by building multiple decision trees during training and outputting the mode of the classes as the prediction.
# The precision score is a metric that measures the proportion of correctly predicted positive cases out of all predicted positive cases.
####################################################################################################

import yfinance as yf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

ticker = 'AAPL'  # Example stock ticker
start_date = '2010-01-01'
end_date = dt.datetime.now()

stock_data = yf.download(ticker, start=start_date, end=end_date)

stock_data['Tomorrow'] = stock_data['Close'].shift(-1)
stock_data['Target'] = (stock_data['Tomorrow'] > stock_data['Close']).astype(int)

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = stock_data['2020-01-01':'2023-04-16']
test = stock_data['2023-04-16':'2024-04-16']

predictors = ['Open', 'High', 'Low', 'Close', 'Volume']

model.fit(train[predictors], train['Target'])

preds = model.predict(test[predictors])

preds = pd.Series(preds, index=test.index, name='Predicted')

# get the precision score
score = precision_score(test['Target'], preds)

print(f"Precision Score: {score:.2f}")

combined = pd.concat([test["Target"], preds], axis=1)

# Print the count of the predicted values
print(combined["Predicted"].value_counts())

combined.plot()
plt.title('Random Forest Classifier - ' + ticker)
plt.show()