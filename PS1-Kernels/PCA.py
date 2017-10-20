#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:09:13 2017

@author: jmarnat
"""
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def PCA(X,n_components):
    # data normalization
    X_centered = X - np.mean(X)
    X_normalized = X_centered / np.std(X)
    
    # covariance matrix
    cov = np.cov(X_normalized.T)
    
    # eigen values & vectors
    eigen_values, eigen_vectors = np.linalg.eig(cov)
    
    # ordering the eigen vectors by decreasing values
    eig_vals_order = np.argsort(eigen_values)[::-1]
    eigen_vectors_decr = eigen_vectors[:,eig_vals_order]

    # creating the n_components PCs over X
    PCs = np.dot(X,eigen_vectors_decr[:,0:n_components])

    return(PCs)


# =============================================================================
# testing the PCA on the MOONS dataset
# =============================================================================

X, y = datasets.make_moons()

plt.scatter(X[y==0,0],X[y==0,1],color='red')
plt.scatter(X[y==1,0],X[y==1,1],color='blue')
plt.show()


# in TWO dimensions
X_new = PCA(X,2)
plt.scatter(x=X_new[y==0,0],y=X[y==0,1],color='blue')
plt.scatter(x=X_new[y==1,0],y=X[y==1,1],color='red')
plt.show()

# in ONE dimension
X_new = PCA(X,1)
plt.scatter(X_new[y==0],np.zeros(len(X_new[y==0])),color='blue',alpha=.5)
plt.scatter(X_new[y==1],np.zeros(len(X_new[y==1])),color='red',alpha=.5)
plt.show()



# =============================================================================
# testing on the IRIS dataset
# =============================================================================

iris = datasets.load_iris()
X = iris.data
y = iris.target

plt.scatter(X[y==0,0],X[y==0,1],color='red')
plt.scatter(X[y==1,0],X[y==1,1],color='blue')
plt.show()


# in ONE dimension
X_new = PCA(X,1)
plt.scatter(X_new[y==0],np.zeros(len(X_new[y==0])),color='blue',alpha=.5)
plt.scatter(X_new[y==1],np.zeros(len(X_new[y==1])),color='red',alpha=.5)
plt.show()


# in TWO dimensions
X_new = PCA(X,2)
plt.scatter(X_new[:,0],X_new[:,1],c=y)
#plt.scatter(x=X_new[y==0,0],y=X[y==0,1],color='blue')
#plt.scatter(x=X_new[y==1,0],y=X[y==1,1],color='red')
plt.show()

X_new = PCA(X,3)
fig = plt.figure()
asub = fig.add_subplot(111,projection='3d')
asub.scatter(X_new[:,0],X_new[:,1],X_new[:,2],c=y,marker='o')
plt.show()












