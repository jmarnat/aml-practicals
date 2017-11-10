#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:39:55 2017

@author: V. Benozillo & J.MARNAT
"""

from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def kmeans_centroids(X, n, eps):
    # n centroids initialization (just once)
    np.random.seed(0)
    r = [np.random.randint(0,len(X)) for i in range(n)]
    
    #todo check r for doubles
    centroids_before = X[r]
    
    while True:
        # 
        y_c = kmeans_fit(X,centroids_before)
        
        centroids_after = np.zeros((n,len(X[0])))
        sum_moove = 0
        for c in range(n):
            for d in range(np.size(X,axis=1)):
                centroids_after[c,d] = np.mean(X[y_c==c,d])
            # adding the distance from the old centroid to the new one
            sum_moove += np.linalg.norm(centroids_before[c] - centroids_after[c])
    
        if (sum_moove <= eps): break
        
        centroids_before = centroids_after
        
    return centroids_after

def kmeans_fit(X,centroids):
    y = np.zeros(len(X))
    dist = np.zeros(len(centroids))
    for i in range(len(X)):
        # computing the argmin distance to each centroid
        for c in range(len(centroids)):
            dist[c] = np.linalg.norm(X[i] - centroids[c])
        y[i] = np.argmin(dist)
        
    return y



def linear(x,y,sigma):
    return np.dot(x,y)

# gaussian rbf kernel
def gaussian(x,y,sigma):
    n = np.linalg.norm(x-y)
    #return np.exp(- (n * n) / sigma)
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


#==============================================================================
# CIRCLES
#==============================================================================
# normal one
np.random.seed(0)
X, y = datasets.make_circles(n_samples=500,random_state=123124, noise=0.0,factor=0.2)

plt.scatter(X[y==0,0],X[y==0,1],color='red',alpha=.5)
plt.scatter(X[y==1,0],X[y==1,1],color='blue',alpha=.5)
plt.show()

# kernelized one
X_centered = X - np.mean(X)
X_normalized = X_centered / np.std(X)
X_normalized = kernel(X_normalized,gaussian,sigma=1)

centers = kmeans_centroids(X_normalized,n=2, eps=0.1)
y_c = kmeans_fit(X_normalized,centers)

plt.scatter(X[y_c==0,0],X[y_c==0,1],color='red',alpha=.5)
plt.scatter(X[y_c==1,0],X[y_c==1,1],color='blue',alpha=.5)
plt.show()


#==============================================================================
# TWO MOONS
#==============================================================================
np.random.seed(0)
X, y = datasets.make_moons(n_samples=500,noise=0.01)

plt.scatter(X[y==0,0],X[y==0,1],color='red')
plt.scatter(X[y==1,0],X[y==1,1],color='blue')
plt.show()

X_centered = X - np.mean(X)
X_normalized = X_centered / np.std(X)
X_normalized = kernel(X_normalized,gaussian,sigma=1)

centers = kmeans_centroids(X_normalized,n=2, eps=0.1)
y_c = kmeans_fit(X_normalized,centers)

#xmin = X_normalized[:,0].min()
#xmax = X_normalized[:,0].max()
#ymin = X_normalized[:,1].min()
#ymax = X_normalized[:,1].max()
#h = .01
#xx,yy = np.meshgrid(np.arange(xmin,xmax,h), np.arange(ymin,ymax,h))
#Z2d = kmeans_fit(np.c_[xx.ravel(),yy.ravel()],centers)
#Z2d = Z2d.reshape(xx.shape)

plt.scatter(X[y_c==0,0],X[y_c==0,1],color='red',alpha=.5)
plt.scatter(X[y_c==1,0],X[y_c==1,1],color='blue',alpha=.5)
#plt.pcolormesh(xx,yy,Z2d,cmap=plt.cm.Paired)
#plt.scatter(X[:,0],X[:,1], c=y_c, cmap=plt.cm.coolwarm)
plt.show()






#==============================================================================
# SWISS ROLL
#==============================================================================

X, y = datasets.make_swiss_roll(n_samples=500,random_state=123124, noise=0.0)
















