#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Data Install
veriler = pd.read_csv('Wine.txt')
X = veriler.iloc[:,0:13].values
y = veriler.iloc[:,13].values


#Data Split to Train And Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


#Data scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(X_train)
X_train2 = pca.transform(X_train)
X_test2 = pca.transform(X_test)


#LR
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

classifier2 = LogisticRegression(random_state = 0)
classifier2.fit(X_train2, y_train)


#Predicts
y_pred = classifier.predict(X_test)

y_pred2 = classifier2.predict(X_test2)


#Confusion matrix
from sklearn.metrics import confusion_matrix
#WİRHOUT PCA
print("WİTHOUT PCA")
cm = confusion_matrix(y_test, y_pred)
print(cm)


#WİTH PCA
print("WİTH PCA")
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)


#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)

X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)


#LR
classifier_lda = LogisticRegression(random_state = 0)
classifier_lda.fit(X_train_lda, y_train)


#Predict
y_pred_lda = classifier_lda.predict(X_test_lda)


#WİTH LDA / ORIJINAL
print("LDA AND ORIJINAL")
cm3 =confusion_matrix(y_pred, y_pred_lda)
print(cm3)