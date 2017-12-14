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

# replaginc 'g' and 'b' by -1 et 1
iono_data[34].replace(['g','b'],[-1,1],inplace=True)

target = iono_data[34]
data = iono_data.loc[:, 0:33]

Accuracy1 = []
Accuracy2 = []
Accuracy3 = []
Accuracy4 = []

for n in range(100) : 
    data_train,data_test,target_train,target_test=train_test_split(data,target,test_size=0.1,random_state=random.seed())
    
    # Anti SVM bug
    if (target_train.values[0]==target_train.values[1]) :
        i = 2
        while (target_train.values[i] == target_train.values[0]) :
            i += 1
            
        sv = target_train.values[1]
        target_train.values[1] = target_train.values[i]
        target_train.values[i] = sv
    
    #==============================================================================
    # classic update on ionosphere
    #==============================================================================
    w = np.zeros(34)
    
    accuracy = []
    
    for i in range(len(data_train)) :
        
        w = update_classic(data_train.values[i], target_train.values[i], w)
        
        acc = 0
        for j in range(len(data_test)):
            if (predict_y(w,data_test.values[j]) == target_test.values[j]):
                acc += 1
        acc /= len(data_test)
        accuracy.append(acc)
        
    Accuracy1.append(accuracy)
    
    #==============================================================================
    # relaxation1 update on ionosphere
    #==============================================================================
    w = np.zeros(34)
    
    accuracy = []
    
    for i in range(len(data_train)) :
        
        w = update_relaxation1(data_train.values[i], target_train.values[i], w, 0.1)
        
        acc = 0
        for j in range(len(data_test)):
            if (predict_y(w,data_test.values[j]) == target_test.values[j]):
                acc += 1
        acc /= len(data_test)
        accuracy.append(acc)
        
    Accuracy2.append(accuracy)
    
    
    #==============================================================================
    # relaxation2 update on ionosphere
    #==============================================================================
    w = np.zeros(34)
    
    accuracy = []
    
    for i in range(len(data_train)) :
        
        w = update_relaxation2(data_train.values[i], target_train.values[i], w, 0.1)
        
        acc = 0
        for j in range(len(data_test)):
            if (predict_y(w,data_test.values[j]) == target_test.values[j]):
                acc += 1
        acc /= len(data_test)
        accuracy.append(acc)
        
    Accuracy3.append(accuracy)
    
    
    #==============================================================================
    # SVM on ionosphere
    #==============================================================================
    accuracy = [0]
    
    X = list(data_train.values[:2])
    y = list(target_train.values[:2])
    
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
    
    Accuracy4.append(accuracy)
    
    
    
accuracy = []
for i in range(len(Accuracy1[0])) :
    moy = 0
    
    for x in Accuracy1 :
        moy += x[i]
        
    accuracy.append(moy/len(Accuracy1))

plt.plot(accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Nb Examples')
plt.show()
    
accuracy = []
for i in range(len(Accuracy2[0])) :
    moy = 0
    
    for x in Accuracy2 :
        moy += x[i]
        
    accuracy.append(moy/len(Accuracy2))

plt.plot(accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Nb Examples')
plt.show()  
    
accuracy = []
for i in range(len(Accuracy3[0])) :
    moy = 0
    
    for x in Accuracy3 :
        moy += x[i]
        
    accuracy.append(moy/len(Accuracy3))

plt.plot(accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Nb Examples')
plt.show()
    
accuracy = []
for i in range(len(Accuracy4[0])) :
    moy = 0
    
    for x in Accuracy4 :
        moy += x[i]
        
    accuracy.append(moy/len(Accuracy4))

plt.plot(accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Nb Examples')
plt.show()   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    