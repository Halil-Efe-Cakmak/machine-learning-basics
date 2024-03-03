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


#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)


#Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)


#visualization
plt.scatter(X, Y ,color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()


#Predicts
print('11 ->', lin_reg.predict([[11]]))
print('6.6 ->', lin_reg.predict([[6.6]]))

print('11 ->', lin_reg2.predict(poly_reg.fit_transform([[11]])))
print('6.6 ->', lin_reg2.predict(poly_reg.fit_transform([[6.6]])))