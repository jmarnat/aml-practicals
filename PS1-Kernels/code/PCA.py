#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:09:13 2017

@author: V. Benozillo & J.MARNAT
"""
from mpl_toolkits.mplot3d import Axes3D
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













