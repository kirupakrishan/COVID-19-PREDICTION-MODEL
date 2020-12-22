# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 20:00:02 2020

@author: Kirupa Krishan
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
import statsmodels.api as sm
dataset = pd.read_csv("Severe_mild.csv")


X = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values

dataset = shuffle(dataset)

# Splitting the dataset into the Training set and Test set
 from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)


# ==========================THE BEST TILL NOW ===================================================

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(tol=1e-4,C=1,penalty='l2',solver='saga',class_weight=None,random_state=0)
regressor = LogisticRegression()
regressor.fit(X_train,y_train)


#X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


y_pred = regressor.predict(X_test)


# Making the Confusion Matrix
y_pred = (y_pred>0.5)
from sklearn.metrics import confusion_matrix,accuracy_score 
cm = confusion_matrix(y_test, y_pred)
precision = accuracy_score(y_test, y_pred)

print(cm)
print("\n")
print(precision)
print("\n")
