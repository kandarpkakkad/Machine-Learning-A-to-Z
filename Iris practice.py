import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

dataset = pd.read_csv("iris.csv")
#print(dataset)

x = dataset.iloc[:, [0]]
y = dataset.iloc[:, [3]]

kmeans = KMeans(n_clusters = 3, max_iter=300, n_init=10, init = 'k-means++',random_state=0)
y_kmeans = kmeans.fit_predict(x,y)

plt.scatter(x[y_kmeans == 0], y[y_kmeans == 0], s = 100, c = 'red', label = 'VERSICOLOR')
plt.scatter(x[y_kmeans == 1], y[y_kmeans == 1], s = 100, c = 'green', label = 'SETOSA')
plt.scatter(x[y_kmeans == 2], y[y_kmeans == 2], s = 100, c = 'blue', label = 'VIRGINIA')
#plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,0], s = 300, c = 'yellow', label = 'Centroids')
plt.title("Flower Classification")
plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.legend()
plt.show()
