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
    #return np.exp(- (n * n) / sigma)
    return np.exp(-sigma * n * n)

def poly(x,y,sigma):
    return 

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
    # centering
    X_centered = X - np.mean(X)
    # normalizing 
    X_normalized = X_centered / np.std(X)

    
    # should make a infinite symbol
    X_normalized = kernel(X_normalized,gaussian,15)
        
    
    # covariance matrix
    cov = np.cov(X_normalized.T)
    
    # eigen values & vectors
    # which are NOT ALWAYS ordered descreasingly
    eigen_values, eigen_vectors = np.linalg.eig(cov)
    
    # indexes of the columns ordered by descresing
    eig_vals_order = np.argsort(eigen_values)[::-1]
    
    eigen_vectors = eigen_vectors[:,eig_vals_order]
   
    
    # PCA in ONE dimension
    X_new = np.dot(X_normalized,eigen_vectors[:,0])
    plt.scatter(x=X_new[y==0],y=np.ones(len(X_new[y==0])),color='blue')
    plt.scatter(x=X_new[y==1],y=np.ones(len(X_new[y==1])),color='red')
    plt.show()
    
    
    # PCA in TWO dimensions
    X_new = np.dot(X_normalized,eigen_vectors[:,0:2])    
    plt.scatter(x=X_new[y==0,0],y=X_new[y==0,1],color='blue')
    plt.scatter(x=X_new[y==1,0],y=X_new[y==1,1],color='red')
    plt.show()
    
    
    # PCA in THREE dimensions
    X_new = np.dot(X_normalized,eigen_vectors[:,0:3])
    fig = plt.figure()
    asub = fig.add_subplot(111,projection='3d')
    ax = X_new[y==0,0]
    bx = X_new[y==0,1]
    cx = X_new[y==0,2]
    ay = X_new[y==1,0]
    by = X_new[y==1,1]
    cy = X_new[y==1,2]
    asub.scatter(ax,bx,cx,c='blue',marker='o')
    asub.scatter(ay,by,cy,c='red',marker='o')
    plt.show()


#
#



#X, y = datasets.make_moons(n_samples=100)
X, y = datasets.make_circles(n_samples=100)

plt.scatter(X[y==0,0],X[y==0,1],color='red')
plt.scatter(X[y==1,0],X[y==1,1],color='blue')
plt.show()

k_pca(X,y,gaussian,15)





