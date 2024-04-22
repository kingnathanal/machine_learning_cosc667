# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Load the Iris dataset
iris = datasets.load_iris()
# Select the first two features
X = iris.data[:, :2]
y = iris.target
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Scale the features
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)
# Define the kernels and polynomial degrees for model training
kernels = {'linear': SVC(kernel='linear'), 'rbf': SVC(kernel='rbf'), 'poly2': SVC(kernel='poly', degree=2), 'poly5': SVC(kernel='poly', degree=5), 'poly10': SVC(kernel='poly', degree=10)}
# Train SVM models
for name, model in kernels.items():
    model.fit(X_train_std, y_train)
# Evaluate each model
for name, model in kernels.items():
    y_pred = model.predict(X_test_std)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of {name}: {accuracy:.2f}')

# Function to plot decision boundaries
def plot_decision_boundaries(X, y, model, title):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(title)
    plt.show()

plot_decision_boundaries(X_test_std, y_test, kernels['linear'], "Linear Kernel")
plot_decision_boundaries(X_test_std, y_test, kernels['rbf'], "RBF Kernel")
plot_decision_boundaries(X_test_std, y_test, kernels['poly2'], "Polynomial Kernel (Degree 2)")
plot_decision_boundaries(X_test_std, y_test, kernels['poly5'], "Polynomial Kernel (Degree 5)")
plot_decision_boundaries(X_test_std, y_test, kernels['poly10'], "Polynomial Kernel (Degree 10)")