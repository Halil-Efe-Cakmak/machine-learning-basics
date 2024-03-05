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


#Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state = 0)
dt.fit(X, Y)


#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)
rf_reg.fit(X, Y.ravel())


#visualization
plt.scatter(X, Y, color = 'red')
plt.plot(x, dt.predict(X), color = 'blue')


#Predict
print(dt.predict([[11]]))
print(dt.predict([[6.6]]))


#R2
from sklearn.metrics import r2_score
print('Decision Tree R2 Score')
print(r2_score(Y, dt.predict(X)))