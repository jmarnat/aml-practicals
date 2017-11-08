#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:19:15 2017

@author: koolok
"""

from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import kPCA
import time
from PCA import PCA

# =============================================================================
# k-PCA over MOONS
# =============================================================================

np.random.seed(0)
X, y = datasets.make_moons(n_samples=500,noise=0.0)

plt.scatter(X[y==0,0],X[y==0,1],color='red')
plt.scatter(X[y==1,0],X[y==1,1],color='blue')
plt.show()


starting_point_proc = time.clock()
PCs = kPCA.k_pca(X,y,kPCA.gaussian,15)
print ("Processing time (moons gaussian 15) : ",time.clock() - starting_point_proc)

starting_point_proc = time.clock()
PCs = kPCA.k_pca(X,y,kPCA.poly,10)
print ("Processing time (moons poly 10) : ",time.clock() - starting_point_proc)

# =============================================================================
# testing the PCA on the MOONS dataset
# =============================================================================

starting_point_proc = time.clock()

# in ONE dimension
X_new = PCA(X,1)
plt.scatter(X_new[y==0],np.zeros(len(X_new[y==0])),color='blue',alpha=.5)
plt.scatter(X_new[y==1],np.zeros(len(X_new[y==1])),color='red',alpha=.5)
plt.show()

# in TWO dimensions
X_new = PCA(X,2)
plt.scatter(x=X_new[y==0,0],y=X[y==0,1],color='blue')
plt.scatter(x=X_new[y==1,0],y=X[y==1,1],color='red')
plt.show()

# in Three dimensions
X_new = PCA(X,3)
plt.scatter(x=X_new[y==0,0],y=X[y==0,1],color='blue')
plt.scatter(x=X_new[y==1,0],y=X[y==1,1],color='red')
plt.show()

print ("Processing time for PCA : ",time.clock() - starting_point_proc)

# =============================================================================
# k-PCA over CIRCLES
# =============================================================================

np.random.seed(0)
X, y = datasets.make_circles(n_samples=500,random_state=123124, noise=0.0,factor=0.2)

plt.scatter(X[y==0,0],X[y==0,1],color='red',alpha=.5)
plt.scatter(X[y==1,0],X[y==1,1],color='blue',alpha=.5)
plt.show()

starting_point_proc = time.clock()
PCs = kPCA.k_pca(X,y,kPCA.gaussian,5)
print ("Processing time (circles gaussian 5) : ",time.clock() - starting_point_proc)

starting_point_proc = time.clock()
PCs = kPCA.k_pca(X,y,kPCA.poly,5)
print ("Processing time (circles poly 5) : ",time.clock() - starting_point_proc)









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
