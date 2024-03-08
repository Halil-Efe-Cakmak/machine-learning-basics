#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Data Install
veriler = pd.read_csv('veriler.txt')

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values


#Data Split to Train And Test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)

#Data scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state = 0)
logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)


#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)