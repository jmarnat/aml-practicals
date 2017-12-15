#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:49:05 2017

@author: koolok
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from update import update_classic, update_relaxation1, update_relaxation2, predict_y

#==============================================================================
# Moons
#==============================================================================
X, y = datasets.make_moons(n_samples=1000)
y = y*2-1

indices = list(range(len(X)))
random.shuffle(indices)

train_indices = indices[0:900]
test_indices = indices[900:]



#==============================================================================
# classic update on moons
#==============================================================================
w = np.zeros(len(X[0]))

t = 0
for i in train_indices :
    t += 1
    
    w = update_classic(X[i], y[i], w)
    
    acc = 0
    for j in test_indices:
        if (predict_y(w,X[j]) == y[j]):
            acc += 1
    acc /= len(test_indices)
    
    if (t == 100) :
        t = 0
        print(acc)
    
#==============================================================================
# relaxation1 update on moons
#==============================================================================
w = np.zeros(len(X[0]))

t = 0
for i in train_indices :
    t += 1
    
    w = update_relaxation1(X[i], y[i], w,0.1)
    
    acc = 0
    for j in test_indices:
        if (predict_y(w,X[j]) == y[j]):
            acc += 1
    acc /= len(test_indices)
    
    if (t == 100) :
        t = 0
        print(acc)   
        
#==============================================================================
# relaxation1 update on moons
#==============================================================================
w = np.zeros(len(X[0]))

t = 0
for i in train_indices :
    t += 1
    
    w = update_relaxation2(X[i], y[i], w,0.1)
    
    acc = 0
    for j in test_indices:
        if (predict_y(w,X[j]) == y[j]):
            acc += 1
    acc /= len(test_indices)
    
    if (t == 100) :
        t = 0
        print(acc) 


#==============================================================================
# the same for without test train
#==============================================================================
w = np.zeros(len(X[0]))
for t in range(len(X)):
    w = update_classic(X[t], y[t], w)

acc = 0
y_predict = []
for i in range(len(X)):
    y_predict.append(predict_y(w,X[i]))
    if (predict_y(w,X[i]) == y[i]):
        acc += 1
y_predict = np.array(y_predict)
acc /= len(X)

plt.scatter(x=X[y_predict==+1,0],y=X[y_predict==+1,1],color='red',alpha=.5)
plt.scatter(x=X[y_predict==-1,0],y=X[y_predict==-1,1],color='blue',alpha=.5)
print('acc = ', acc)


