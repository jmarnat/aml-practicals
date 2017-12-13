#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:55:04 2017

@author: koolok
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np
from update import update_classic, update_relaxation1, update_relaxation2, predict_y
import matplotlib.pyplot as plt
from sklearn import svm

iono_data = pd.read_csv("ionosphere.data",header=None)

target = iono_data[34]
data = iono_data.loc[:, 0:33]

data_train,data_test,target_train,target_test=train_test_split(data,target,test_size=0.1,random_state=random.seed())

#==============================================================================
# classic update on ionosphere
#==============================================================================
w = np.zeros(len(data_train.ix[0]))

accuracy = []

for i in range(len(data_train)) :
    
    w = update_classic(data_train.values[i], target_train.values[i], w)
    
    acc = 0
    for j in range(len(data_test)):
        if (predict_y(w,data_test.values[j]) == target_test.values[j]):
            acc += 1
    acc /= len(data_test)
    accuracy.append(acc)
    
plt.plot(accuracy)
plt.ylabel('Accuracy')
plt.show()

#==============================================================================
# relaxation1 update on ionosphere
#==============================================================================
w = np.zeros(len(data_train.ix[0]))

accuracy = []

for i in range(len(data_train)) :
    
    w = update_relaxation1(data_train.values[i], target_train.values[i], w, 0.1)
    
    acc = 0
    for j in range(len(data_test)):
        if (predict_y(w,data_test.values[j]) == target_test.values[j]):
            acc += 1
    acc /= len(data_test)
    accuracy.append(acc)
    
plt.plot(accuracy)
plt.ylabel('Accuracy')
plt.show()


#==============================================================================
# relaxation2 update on ionosphere
#==============================================================================
w = np.zeros(len(data_train.ix[0]))

accuracy = []

for i in range(len(data_train)) :
    
    w = update_relaxation2(data_train.values[i], target_train.values[i], w, 0.1)
    
    acc = 0
    for j in range(len(data_test)):
        if (predict_y(w,data_test.values[j]) == target_test.values[j]):
            acc += 1
    acc /= len(data_test)
    accuracy.append(acc)
    
plt.plot(accuracy)
plt.ylabel('Accuracy')
plt.show()


#==============================================================================
# SVM on ionosphere
#==============================================================================
accuracy = []

X = list(data_train.values[:5])
y = list(target_train.values[:5])

clf = svm.SVC()
clf.fit(X, y)

results = clf.predict(data_test.values)

acc = 0
for j in range(len(results)) :
    if (results[j] == target_test.values[j]) :
        acc += 1
    
acc /= len(data_test)
accuracy.append(acc)

for i in range(5,len(data_train)) :
    X.append(data_train.values[i])
    y.append(target_train.values[i])

    clf = svm.SVC()
    clf.fit(X, y)
    
    results = clf.predict(data_test.values)
    
    acc = 0
    for j in range(len(results)) :
        if (results[j] == target_test.values[j]) :
            acc += 1
        
    acc /= len(data_test)
    accuracy.append(acc)

    
plt.plot(accuracy)
plt.ylabel('Accuracy')
plt.show()