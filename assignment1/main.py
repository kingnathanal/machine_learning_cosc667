# pip install pandas
# pip install scikit-learn
# pip install matplotlib

import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree

# Load the dataset
file_path = 'car_evaluation_noisy.csv' # Replace with the actual file path
data = pd.read_csv(file_path)
# Task 1: Data Cleaning
# Remove rows with missing values
# ...
cleaned_data = data.dropna()

for x in cleaned_data.index:
    if cleaned_data.loc[x, 'persons'] == 'more':
        cleaned_data.loc[x, 'persons'] = '5'
    if cleaned_data.loc[x, 'doors'] == '5more':
        cleaned_data.loc[x, 'doors'] = '5'

# Encode categorical variables
label_encoders = {}
for column in cleaned_data.columns[:1]: # Assuming the last column is the target 
    le = LabelEncoder()
    cleaned_data[column] = le.fit_transform(cleaned_data[column])
    label_encoders[column] = le

# Separate features and target variable
# ...
X = cleaned_data.drop(columns='buying').copy()
X = pd.get_dummies(X, columns=['maint','lug_boot','safety','class','doors','persons'])
y = cleaned_data['buying'].copy()
# Task 2: Decision Tree Model
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# Create and train the decision tree model
# Note: Ensure that the criterion is set to 'gini'
# ...
clf = DecisionTreeClassifier(criterion="gini", max_depth=6, random_state=1)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
# Task 3: Extracting Decision Tree Information
# Your code to extract and print the decision tree node information goes here
# ...
plt.figure(figsize=(20,20),dpi=300)
plot_tree(clf, feature_names=X.columns, filled=True, precision = 4, rounded = True)
plt.show()