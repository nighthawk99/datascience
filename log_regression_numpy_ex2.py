# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 18:48:46 2018

@author: micha
"""

import numpy as np # imports a fast numerical programming library
import scipy as sp #imports stats functions, amongst other things
import matplotlib as mpl # this actually imports matplotlib
import matplotlib.cm as cm #allows us easy access to colormaps
import matplotlib.pyplot as plt #sets up plotting under plt
import pandas as pd #lets us handle data as dataframes
import math

# Exercise 2

def costFunction_reg(theta,x,y,lam):
    ssize = df_x.shape[0]
    size_theta = theta.shape[0]
    cF = (-y.dot(np.log(sigmoid(x.dot(theta)))) - (1-y).dot(np.log(1-sigmoid(x.dot(theta))))
    + lam/2*theta[1:].dot(theta[1:])) /ssize
    x_split = np.hsplit(x, size_theta)
    x_split = np.array([np.reshape(x_i,ssize) for x_i in x_split])
    grad = x_split.dot((sigmoid(x.dot(theta)) - y)) /ssize + lam/ssize * np.insert(theta[1:],0,0)
    return [cF, grad]

df_x = pd.read_csv("data/ex2/ex2data2.csv", sep=",", usecols = [0,1], header=None, engine='python')
df_y = pd.read_csv('data/ex2/ex2data2.csv', sep=",", usecols = [2], header=None, engine='python')
df_y = df_y.astype(int)
df_x.head()

input_x_scale =1
x = df_x.values
x = x = x/input_x_scale
x = np.hstack([np.ones((df_x.shape[0], 1)), x]) #values return numpy version of panda
y = df_y[df_y.columns.values[0]].values

for p in range(1,len(y)):
  if y[p-1]==0:
    plt.plot(x[p-1][1],x[p-1][2], color='green', marker = 'o', linestyle = 'none') 
  else:
    plt.plot(x[p-1][1],x[p-1][2], color='red', marker = 'o', linestyle = 'none')
plt.show()

x_extended = []
for x_pair in x:
    a = x_extended.append(np.array([math.pow(x_pair[1],j) * math.pow(x_pair[2],i-j) for i in range(0,7) for j in range(0,i+1)]))
np.asarray(x_extended)    

