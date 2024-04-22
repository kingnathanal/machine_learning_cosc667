# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Load the Iris dataset
iris = datasets.load_iris()
# Select the first two features
df = pd.DataFrame(iris.data, columns=iris.feature_names)
selected_features = df[['sepal length (cm)', 'sepal width (cm)']]

print(selected_features.head())
# Select the first two features
#X = iris.data[:, :2]

# Select the target
#y = iris.target