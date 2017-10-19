#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:09:13 2017

@author: jmarnat
"""
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


X, y = datasets.make_moons()

plt.scatter(X[y==0,0],X[y==0,1],color='red')
plt.scatter(X[y==1,0],X[y==1,1],color='blue')
plt.show()


# data normalization
# centering
X_centered = X - np.mean(X)
# normalizing 
X_normalized = X_centered / np.std(X)

# check:
print(np.mean(X_normalized))
print(np.std(X_normalized))
    
# covariance matrix
cov = np.cov(X_normalized.T)

# eigen values & vectors
eigen_values, eigen_vectors = np.linalg.eig(cov)

# indexes of the columns ordered by descresing
eig_vals_order = np.argsort(eigen_values)[::-1]

eigen_vals_decr = eigen_values[eig_vals_order]
eigen_vectors_decr = eigen_vectors[:,eig_vals_order]

tot_eigen = sum(eigen_values)

var_exp = [(i / tot_eigen)*100 for i in sorted(eigen_values, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.scatter(range(1,3),var_exp)
plt.scatter(range(1,3),np.cumsum(var_exp))
plt.show()


# plotting X before PCA
plt.scatter(X[y==0,0],X[y==0,1],color='blue')
plt.scatter(X[y==1,0],X[y==1,1],color='red')
plt.show()

# applying the PCA in TWO dimensions
X_new = np.dot(X,eigen_vectors_decr[:,:])

plt.scatter(x=X_new[y==0,0],y=X[y==0,1],color='blue')
plt.scatter(x=X_new[y==1,0],y=X[y==1,1],color='red')
plt.show()

# in ONE dimension
X_new = np.dot(X,eigen_vectors_decr[:,0])
plt.scatter(X_new[y==0],np.zeros(len(X_new[y==0])),color='blue',alpha=.5)
plt.scatter(X_new[y==1],np.zeros(len(X_new[y==1])),color='red',alpha=.5)
plt.show()



















