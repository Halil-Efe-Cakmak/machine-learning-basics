#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Data Install
veriler =pd.read_csv('musteriler.txt')

X = veriler.iloc[:,3:].values


#Hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
Y_predict = ac.fit_predict(X)
print(Y_predict)

#visualization
plt.scatter(X[Y_predict == 0, 0], X[Y_predict == 0, 1], c = 'red')
plt.scatter(X[Y_predict == 1, 0], X[Y_predict == 1, 1], c = 'blue')
plt.scatter(X[Y_predict == 2, 0], X[Y_predict == 2, 1], c = 'green')
plt.scatter(X[Y_predict == 3, 0], X[Y_predict == 3, 1], c = 'yellow')
plt.title('Hierarchical clustering')
plt.show()


#Dendrogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.show()


