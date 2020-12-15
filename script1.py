# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Severe_mild.csv')
X = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
#THIS IS TO FING THE OPTIMAL NUMBER OF CLUSTERS
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()


  # Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)



# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 2, init = 'k-means++')
y_kmeans=kmeans.fit_predict(X)
#y_pred = kmeans.predict(X_test)



# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix,accuracy_score 
# cm = confusion_matrix(y_test, y_pred)
# precision = accuracy_score(y_test, y_pred)

# print(cm)
# print("\n")
# print(precision)
# print("\n")

