import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#load wholesale data using pandas
wholesale = pd.read_csv('wholesale-customers-data.csv')
cleaned_wholesale_data = wholesale.dropna()

# Drop the 'Channel' and 'Region' category columns
wholesale_normalized = cleaned_wholesale_data.drop(['Channel','Region'], axis=1).copy()

# select features and targets
features = wholesale_normalized
targets = cleaned_wholesale_data['Region']

X = features
y = targets

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# initial use of PCA before determining the number of components to retain
pca = PCA()
pca.fit(X_std)

explained_variance = pca.explained_variance_ratio_

# Plotting the Scree Plot
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, color='blue')
plt.title('Scree Plot')
plt.xlabel('Number of Components')
plt.ylabel('Variance Explained')
plt.show()

# only 2 components are enough to explain the variance
pca = PCA(n_components=2)
componenets = pca.fit_transform(X_std)

# Create an elbow plot
inertia = []

for i in range(1, 10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(componenets)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), inertia, marker='o')
plt.title('Elbow Plot')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Clustering
kmeans = KMeans(n_clusters=5)
cluster_labels = kmeans.fit_predict(X_std)

plt.figure(figsize=(8, 5))
plt.scatter(componenets[:, 0], componenets[:, 1], c=cluster_labels)
plt.title('Clusters of Customers')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='o', s=200)

plt.show()


