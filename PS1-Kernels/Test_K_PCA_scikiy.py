#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:08:29 2017

@author: koolok
"""

from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_moons, make_circles
import matplotlib.pyplot as plt
import numpy as np

X,y = make_circles()

plt.title("Original Circles")
plt.scatter(X[y==0, 0], X[y==0, 1], color='red')
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue')
plt.savefig("results_val/original_circles.jpg")
plt.show()


# PCA, n_components = 1
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)

plt.title("PCA, n_components = 1")
plt.scatter(X_pca[y==0, 0], np.zeros((50,1)), color='red', alpha=0.5)
plt.scatter(X_pca[y==1, 0], np.zeros((50,1)), color='blue', alpha=0.5)
plt.savefig("results_val/PCA_n=1.jpg")
plt.show()

# K-PCA, linear, n_components = 1
scikit_kpca = KernelPCA(n_components=1, kernel='linear')
X_skernpca = scikit_kpca.fit_transform(X)

plt.title("K-PCA, linear, n_components = 1")
plt.scatter(X_skernpca[y==0, 0], np.zeros((50,1)), color='red', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], np.zeros((50,1)), color='blue', alpha=0.5)
plt.savefig("results_val/K-PCA_linear_n=1.jpg")
plt.show()

# K-PCA, poly, n_components = 1
for i in range(20) :
    scikit_kpca = KernelPCA(n_components=1, kernel='poly', degree=i)
    X_skernpca = scikit_kpca.fit_transform(X)
    
    plt.title("K-PCA, poly, degree="+str(i)+", n_components = 1")
    plt.scatter(X_skernpca[y==0, 0], np.zeros((50,1)), color='red', alpha=0.5)
    plt.scatter(X_skernpca[y==1, 0], np.zeros((50,1)), color='blue', alpha=0.5)
    plt.savefig("results_val/poly1/"+str(i)+".jpg")
    plt.show()

# K-PCA, sigmoid, n_components = 1
scikit_kpca = KernelPCA(n_components=1, kernel='sigmoid')
X_skernpca = scikit_kpca.fit_transform(X)

plt.title("K-PCA, sigmoid, n_components = 1")
plt.scatter(X_skernpca[y==0, 0], np.zeros((50,1)), color='red', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], np.zeros((50,1)), color='blue', alpha=0.5)
plt.savefig("results_val/K-PCA_sigmoid_n=1.jpg")
plt.show()

# K-PCA, cosine, n_components = 1
scikit_kpca = KernelPCA(n_components=1, kernel='cosine')
X_skernpca = scikit_kpca.fit_transform(X)

plt.title("K-PCA, cosine, n_components = 1")
plt.scatter(X_skernpca[y==0, 0], np.zeros((50,1)), color='red', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], np.zeros((50,1)), color='blue', alpha=0.5)
plt.savefig("results_val/K-PCA_cosine_n=1.jpg")
plt.show()

# K-PCA, rbf, n_components = 1
for i in range(20) :
    scikit_kpca = KernelPCA(n_components=1, kernel='rbf', gamma=i)
    X_skernpca = scikit_kpca.fit_transform(X)
    
    plt.title("K-PCA, rbf, gamma="+str(i)+", n_components = 1")
    plt.scatter(X_skernpca[y==0, 0], np.zeros((50,1)), color='red', alpha=0.5)
    plt.scatter(X_skernpca[y==1, 0], np.zeros((50,1)), color='blue', alpha=0.5)
    plt.savefig("results_val/rbf/K-PCA_rbf="+str(i)+".jpg")
    plt.show()


# K-PCA, laplacian, n_components = 1
for i in range(30) :
    scikit_kpca = KernelPCA(n_components=1, kernel='laplacian', gamma=i)
    X_skernpca = scikit_kpca.fit_transform(X)
    
    plt.title("K-PCA, laplacian, gamma="+str(i)+", n_components = 1")
    plt.scatter(X_skernpca[y==0, 0], np.zeros((50,1)), color='red', alpha=0.5)
    plt.scatter(X_skernpca[y==1, 0], np.zeros((50,1)), color='blue', alpha=0.5)
    plt.savefig("results_val/laplacian/"+str(i)+".jpg")
    plt.show()

# PCA, n_components = 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.title("PCA, n_components = 2")
plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], color='red')
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='blue')
plt.savefig("results_val/PCA_n=2.jpg")
plt.show()


# K-PCA, rbf, n_components = 2
for i in range(20) :
    scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=i)
    X_skernpca = scikit_kpca.fit_transform(X)
    
    plt.title("K-PCA, rbf, gamma="+str(i)+", n_components = 2")
    plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], color='red', alpha=0.5)
    plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], color='blue', alpha=0.5)
    plt.savefig("results_val/rbf2/K-PCA_rbf="+str(i)+".jpg")
    plt.show()


# K-PCA, sigmoid, n_components = 2
scikit_kpca = KernelPCA(n_components=2, kernel='sigmoid')
X_skernpca = scikit_kpca.fit_transform(X)

plt.title("K-PCA, sigmoid, n_components = 2")
plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], color='red', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], color='blue', alpha=0.5)
plt.savefig("results_val/K-PCA_sigmoid_n=2.jpg")
plt.show()

# K-PCA, linear, n_components = 2
scikit_kpca = KernelPCA(n_components=2, kernel='linear')
X_skernpca = scikit_kpca.fit_transform(X)

plt.title("K-PCA, linear, n_components = 2")
plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], color='red', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], color='blue', alpha=0.5)
plt.savefig("results_val/K-PCA_linear_n=1.jpg")
plt.show()

# K-PCA, laplacian, n_components = 2
for i in range(30) :
    scikit_kpca = KernelPCA(n_components=2, kernel='laplacian', gamma=i)
    X_skernpca = scikit_kpca.fit_transform(X)
    
    plt.title("K-PCA, laplacian, gamma="+str(i)+", n_components = 1")
    plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], color='red', alpha=0.5)
    plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], color='blue', alpha=0.5)
    plt.savefig("results_val/laplacian2/"+str(i)+".jpg")
    plt.show()

# K-PCA, poly, n_components = 2
for i in range(20) :
    scikit_kpca = KernelPCA(n_components=2, kernel='poly', degree=i)
    X_skernpca = scikit_kpca.fit_transform(X)
    
    plt.title("K-PCA, poly, degree="+str(i)+", n_components = 2")
    plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], color='red', alpha=0.5)
    plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], color='blue', alpha=0.5)
    plt.savefig("results_val/poly2/"+str(i)+".jpg")
    plt.show()
    
# K-PCA, cosine, n_components = 2
scikit_kpca = KernelPCA(n_components=2, kernel='cosine')
X_skernpca = scikit_kpca.fit_transform(X)

plt.title("K-PCA, cosine, n_components = 2")
plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], color='red', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], color='blue', alpha=0.5)
plt.savefig("results_val/K-PCA_cosine_n=2.jpg")
plt.show()