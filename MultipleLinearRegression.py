#Libraries
import pandas as pd
import numpy as np

#Data Install
veriler = pd.read_csv('veriler.txt')

Yas = veriler.iloc[:,1:4].values


#Encoder: Kategoric -> Numeric
ulke = veriler.iloc[:,0:1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:,0]  = le.fit_transform(ulke[:,0])
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()



c = veriler.iloc[:,-1:].values
le = preprocessing.LabelEncoder()
c[:,-1]  = le.fit_transform(veriler.iloc[:,-1])
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()

                                 
#Transfor Numpy To Dataframe
sonuc = pd.DataFrame(data = ulke, index = range(22), columns = ['fr', 'tr', 'us'])
sonuc2 = pd.DataFrame(data = Yas, index = range(22), columns = ['boy', 'kilo', 'Yas'])
sonuc3 = pd.DataFrame(data = c[:,:1], index = range(22), columns = ['cinsiyet'])
                    

#Combine Dataframes
s = pd.concat([sonuc, sonuc2], axis = 1)
s2 = pd.concat([s, sonuc3], axis = 1)


#Data Split to Train And Test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size = 0.33, random_state = 0)


#Predict cinsiyet
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)


#predict boy
boy = s2.iloc[0:,3:4].values

left = s2.iloc[:,:3]
right = s2.iloc[:,4:]
veri = pd.concat([left, right], axis = 1)

x1_train, x1_test, y1_train, y1_test = train_test_split(veri, boy, test_size = 0.33, random_state = 0)

r2 = LinearRegression()
r2.fit(x1_train, y1_train)
y1_pred = r2.predict(x1_test)


#Backward elemination
import statsmodels.api as sm
X = np.append(arr = np.ones((22,1)).astype(int), values = veri, axis = 1)

X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(boy, X_l).fit()
print(model.summary())

X_l = veri.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(boy, X_l).fit()
print(model.summary())

X_l = veri.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(boy, X_l).fit()
print(model.summary())










