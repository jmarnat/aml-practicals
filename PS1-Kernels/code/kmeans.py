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



# =============================================================================
# test on IRIS
# =============================================================================

iris = datasets.load_iris()
X = iris.data[:,[0,1,2]]
y = iris.target

centers = kmeans_centroids(X,3, 0.1)
y_c = kmeans_fit(X,centers)

plt.scatter(X[:,0],X[:,1],c=y)
plt.title('iris cp 1 & 2')
plt.show()

plt.scatter(X[:,0],X[:,1],c=y_c)
plt.title('iris CLUSTERIZED')
plt.show()

fig = plt.figure()
asub = fig.add_subplot(111,projection='3d')
asub.scatter(X[:,0],X[:,1],X[:,2],c=y,marker='o',alpha=.9)
asub.set_title('iris real classes')
plt.show()

fig = plt.figure()
asub = fig.add_subplot(111,projection='3d')
asub.scatter(X[:,0],X[:,1],X[:,2],c=y_c,marker='o',alpha=.9)
asub.set_title('iris k-means clusters')
plt.show()



# =============================================================================
# test on circles
# =============================================================================

X, y = datasets.make_circles(n_samples=500, noise=0.01)
plt.scatter(X[:,0],X[:,1],c=y,alpha=.5)
plt.title("circles original")
plt.show()



















