import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, [3,4]].values

# Using the dendrogram to find the optimal number of clusters

#dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
#plt.title("Dendrogram")
#plt.xlabel("Customers")
#plt.ylabel("ED")
#plt.show()

# Fitting hierarchical clustering to dataset

ac = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage='ward')
y_pred = ac.fit_predict(x)

# Visualisation of clusters

plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s = 100, c = 'red', label = 'Target')
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s = 100, c = 'green', label = 'Standard')
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], s = 100, c = 'blue', label = 'Regular')
plt.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1], s = 100, c = 'black', label = 'Foolish')
plt.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1], s = 100, c = 'magenta', label = 'Discount')
plt.title("Customer cluster")
plt.xlabel("Annual income")
plt.ylabel("Spending score")
plt.legend()
plt.show()
