#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Data Install
veriler = pd.read_csv('eksikveriler.txt')

#Missin Data Imputer
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
Yas = veriler.iloc[:,1:4].values
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])



#Encoder: Kategoric -> Numeric
ulke = veriler.iloc[:,0:1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing
le = LabelEncoder()
ulke[:,0]  = le.fit_transform(ulke[:,0])
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
                                 
#Transfor Numpy To Dataframe
sonuc = pd.DataFrame(data = ulke, index = range(22), columns = ['fr', 'tr', 'us'])
sonuc2 = pd.DataFrame(data = Yas, index = range(22), columns = ['boy', 'kilo', 'Yas'])
cinsiyet = veriler.iloc[:,-1].values
sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns = ['cinsiyet'])
                    

#Combine Dataframes
s = pd.concat([sonuc, sonuc2], axis = 1)
s2 = pd.concat([s, sonuc3], axis = 1)


#Data Split to Train And Test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size = 0.33, random_state = 0)

#Data scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
