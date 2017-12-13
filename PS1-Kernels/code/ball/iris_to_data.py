#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 17:52:47 2017

@author: jmarnat
"""

from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y == 0,]

export_y = False

file = open('iris.mod','w')

file.write("param X{1..")
file.write(str(len(X)))
file.write(", 1..4};\n")

if (export_y):
	file.write("param Y{1..")
	file.write(str(len(X)))
	file.write("};\n\n")

file.write("param dim;\n")
file.write("param n_feat;\n")

file.write("data;\n")

file.write("param X : 1 2 3 4 :=\n")
for i in range(len(X)):
    file.write(str(i+1))
    for j in range(len(X[0])):
        file.write(" ")
        file.write(str(X[i,j]))
    file.write("\n")
file.write(";\n")

if (export_y):
	file.write("param Y : 1 :=\n")
	for i in range(len(X)):
	    file.write(str(i+1))
	    file.write(" ")
	    file.write(str(y[i]))
	    file.write("\n")
	file.write(";\n")

file.write("param dim = ")
file.write(str(len(X[0])))
file.write("\n")

file.write("param n_feat = ")
file.write(str(len(X)))
file.write("\n")

file.close()

print("done!")
