import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('satislar.txt')
print(veriler)

aylar = veriler[['Aylar']]
satislar = veriler[['Satislar']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size = 0.33, random_state = 0)

#Model Building
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test, tahmin)
plt.title('Aylara göre satış')
plt.xlabel('Aylar')
plt.ylabel('Satışlar')

