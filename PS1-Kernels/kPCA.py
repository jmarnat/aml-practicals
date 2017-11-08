#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:59:13 2017

@author: jmarnat
"""
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


def linear(x,y,sigma):
    return np.dot(x,y)

# gaussian rbf kernel
def gaussian(x,y,sigma):
    n = np.linalg.norm(x-y)
    # return np.exp(- (n * n) / (2 * sigma * sigma))
    return np.exp(-sigma * n * n)

def poly(x,y,sigma):
    d = np.dot(x,y)
    #return np.pow(d,2)
    return (d + 1) * (d + 1)

def kernel(X,function,sigma):
    n = np.size(X,0)
    K = np.zeros([n,n])

    for i in range(n):
        for j in range(n):
            K[i,j] = function(X[i,:],X[j,:],sigma)
    
    # centering K
    Kn = np.ones([n,n])/n
    K = K - Kn * K - K * Kn + Kn * K * Kn
    
    return(K)



def k_pca(X,y,function,sigma):
        
    # data normalization
    X_centered = X - np.mean(X)
    X_normalized = X_centered / np.std(X)
    K = kernel(X_normalized,function,sigma)

    # covariance matrix
    cov = np.cov(K.T)
    
    # eigen values & vectors
    # which are NOT ALWAYS ordered descreasingly
    eigen_values, eigen_vectors = np.linalg.eig(cov)
    eig_vals_order = np.argsort(eigen_values)[::-1]
    eigen_vectors = eigen_vectors[:,eig_vals_order]
       
    # PCA in ONE dimension
    X_new = np.dot(K,eigen_vectors[:,0])
    plt.scatter(x=X_new[y==0],y=np.ones(len(X_new[y==0])),color='blue',alpha=.5)
    plt.scatter(x=X_new[y==1],y=np.ones(len(X_new[y==1])),color='red',alpha=.5)
    plt.show()
    
    
    # PCA in TWO dimensions
    X_new = np.dot(K,eigen_vectors[:,0:2])    
    plt.scatter(x=X_new[y==0,0],y=X_new[y==0,1],color='blue',alpha=.5)
    plt.scatter(x=X_new[y==1,0],y=X_new[y==1,1],color='red',alpha=.5)
    plt.show()
    
    
    # PCA in THREE dimensions
    X_new = np.dot(K,eigen_vectors[:,0:3])
    X_new = X_new.astype('float64')
    fig = plt.figure()
    asub = fig.add_subplot(111,projection='3d')
    ax = X_new[y==0,0]
    bx = X_new[y==0,1]
    cx = X_new[y==0,2]
    ay = X_new[y==1,0]
    by = X_new[y==1,1]
    cy = X_new[y==1,2]
    asub.scatter(ax,bx,cx,c='blue',marker='o',alpha=.5)
    asub.scatter(ay,by,cy,c='red',marker='o',alpha=.5)
    plt.show()
    
    return K




np.random.seed(0)
X, y = datasets.make_moons(n_samples=500,noise=0.01)

plt.scatter(X[y==0,0],X[y==0,1],color='red')
plt.scatter(X[y==1,0],X[y==1,1],color='blue')
plt.show()

PCs = k_pca(X,y,gaussian,10)
PCs = k_pca(X,y,poly,10)



# =============================================================================
# k-PCA over CIRCLES
# =============================================================================

np.random.seed(0)
X, y = datasets.make_circles(n_samples=500,random_state=123124, noise=0.0,factor=0.2)

plt.scatter(X[y==0,0],X[y==0,1],color='red',alpha=.5)
plt.scatter(X[y==1,0],X[y==1,1],color='blue',alpha=.5)
plt.show()

PCs = k_pca(X,y,gaussian,5)
PCs = k_pca(X,y,poly,5)
