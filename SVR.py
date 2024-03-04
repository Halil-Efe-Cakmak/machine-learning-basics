#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Data Install
veriler = pd.read_csv('maaslar.txt')


#Data frame slices
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]


#Numpy array conversion
X = x.values
Y = y.values


#Data scaling
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_scaled = sc1.fit_transform(X)
sc2 = StandardScaler()
y_scaled = np.ravel(sc2.fit_transform(Y.reshape(-1, 1)))


#SVR
from sklearn.svm import SVR
svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_scaled, y_scaled)


#visualization
plt.scatter(x_scaled, y_scaled, color = 'red')
plt.plot(x_scaled, svr_reg.predict(x_scaled), color = 'blue')



#R2
from sklearn.metrics import r2_score
print('SVR R2 Score')
print(r2_score(Y, svr_reg.predict(X)))






