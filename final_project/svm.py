# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Load stock price data
ticker = 'AAPL'  # Example: Apple Inc.
start_date = '2023-03-12'
end_date = '2024-03-12'
data = yf.download(ticker, start=start_date, end=end_date)

data["Return"] = data["Close"].pct_change()
data["Previous Return"] = data["Return"].shift(1)
data["Target"] = (data['Return'] > 0).astype(int)
data = data.dropna()

X = data[["Previous Return"]]
y = data["Target"]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scale the features
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)
# Define the kernels and polynomial degrees for model training
kernels = {'rbf': SVC(kernel='rbf'), 'poly2': SVC(kernel='poly', degree=2), 'poly5': SVC(kernel='poly', degree=5), 'poly10': SVC(kernel='poly', degree=10)}
# Train SVM models
for name, model in kernels.items():
    model.fit(X_train_std, y_train)
# Evaluate each model
for name, model in kernels.items():
    y_pred = model.predict(X_test_std)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of {name}: {accuracy:.2f}')
    print(classification_report(y_test, y_pred))

# function to plot actual returns and predicted returns
def plot_returns(X, y, model, title):
    plt.plot(X, y, color='black', label='Actual data')
    plt.plot(X, model.predict(X), label=f'Predicted by {title}', linewidth=2)
    plt.title(f'{title} Prediction vs Actual')
    plt.xlabel('Previous Return')
    plt.ylabel('Return')
    plt.legend()
    plt.show()

#plot_returns(X_test_std, y_test, kernels['linear'], "Linear Kernel")
plot_returns(X_test_std, y_test, kernels['rbf'], "RBF Kernel")
plot_returns(X_test_std, y_test, kernels['poly2'], "Polynomial Kernel (Degree 2)")
#plot_returns(X_test_std, y_test, kernels['poly5'], "Polynomial Kernel (Degree 5)")
#plot_returns(X_test_std, y_test, kernels['poly10'], "Polynomial Kernel (Degree 10")