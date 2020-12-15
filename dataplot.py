# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 01:22:54 2020

@author: Kushal
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Covid-Survey.csv")

X = dataset.iloc[:,0:11].values
y = dataset.iloc[:,-1].values

plt.scatter(x, y)
plt.show()
