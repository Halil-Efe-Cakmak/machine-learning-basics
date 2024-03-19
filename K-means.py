#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Data Install
veriler =pd.read_csv('musteriler.txt')

X = veriler.iloc[:,3:].values


#K-means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4, init = 'k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)


#WCSS
sonuclar = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
    
plt.plot(range(1, 11), sonuclar)    
    