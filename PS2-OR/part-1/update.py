#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:22:26 2017

@author: koolok
"""

import numpy as np


def update_classic(x,y,w):
    loss = max(0, 1-y*np.dot(w,x))
    tau = loss / (np.linalg.norm(x)**2)
    return w + tau*y*x

def update_relaxation1(x,y,w,c):
    loss = max(0, 1-y*np.dot(w,x))
    tau = min(c,loss / (np.linalg.norm(x)**2))
    return w + tau*y*x

def update_relaxation2(x,y,w,c):
    loss = max(0, 1-y*np.dot(w,x))
    tau = loss / ( (np.linalg.norm(x)**2) + 1/(2*c))
    return w + tau*y*x

def predict_y(w,x):
    return np.sign(np.dot(w,x))