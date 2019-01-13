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

def sigmoid(x):
    return 1/(1+np.exp(-x))

def costFunction_reg(theta,x,y,lam):
    ssize = y.shape[0]
    size_theta = theta.shape[0]
    cF = (-y.dot(np.log(sigmoid(x.dot(theta)))) - (1-y).dot(np.log(1-sigmoid(x.dot(theta))))+ lam/2*(theta[1:]).dot(theta[1:])) /ssize
    x_split = np.hsplit(x, size_theta)
    x_split = np.array([np.reshape(x_i,ssize) for x_i in x_split])
    grad = x_split.dot((sigmoid(x.dot(theta)) - y)) /ssize + lam/ssize * np.insert(theta[1:],0,0)
    return [cF, grad]

def gradientDesc_reg(maxIter, precision, init_theta, x,y,stepmult,lam):
    stepSizes =[]
    prev_stepSize =1
    costF_list = []
    theta_list=[]
    iters = 0
    theta = init_theta
    costF = costFunction_reg(theta,x,y,lam)
    costF_list.append(costF[0])
    theta_list.append(theta)

    while prev_stepSize > precision and iters < maxIter:
        val=costF[0]
        grad = costF[1]
        theta = theta - stepmult * grad
        costF = costFunction_reg(theta,x,y,lam)
        prev_stepSize = abs(costF[0]-val)
        stepSizes.append(prev_stepSize)
        costF_list.append(costF[0])
        theta_list.append(theta)
        iters=iters + 1
    return [costF, theta,iters,stepSizes, costF_list, theta_list]

def mapFeaturePoly(x,deg):
    x_ext = []
    for x_pair in x:
        x_ext.append(np.array([math.pow(x_pair[0],j) * math.pow(x_pair[1],i-j) for i in range(0,deg+1) for j in range(0,i+1)]))
    return np.array(x_ext)
    
lam=0.01 # regularization parameter

df_x = pd.read_csv("data/ex2/ex2data2.csv", sep=",", usecols = [0,1], header=None, engine='python')
df_y = pd.read_csv('data/ex2/ex2data2.csv', sep=",", usecols = [2], header=None, engine='python')
df_y = df_y.astype(int)
df_x.head()
input_x_scale =1
x = df_x.values
x = x = x/input_x_scale
y = df_y[df_y.columns.values[0]].values

x_ext = mapFeaturePoly(x,6)

theta = np.array([0 for x in range(1,x_ext[1].shape[0]+1)])
gD=gradientDesc_reg(1000000,0.00000001,theta, x_ext, y,0.5,lam)

theta = gD[1]

print("Cost funtion: " + str(gD[0][0]))
print("Iterations: " + str(gD[2]))
print("Theta: " + str(gD[1]))   

# plot decision boundary

u = np.linspace(-1,1.5,50)
v = np.linspace(-1,1.5,50)
z = np.zeros((len(u),len(v)))

for i in range(1,len(u)):
    for j in range(1, len(v)):
        z[i][j]=theta.dot(mapFeaturePoly(np.array([np.array((u[i],v[j]))]),6)[0])

plt.contour(u,v,z,levels=[0])
for p in range(1,len(y)):
  if y[p-1]==0:
    plt.plot(x[p-1][0],x[p-1][1], color='green', marker = 'o', linestyle = 'none') 
  else:
    plt.plot(x[p-1][0],x[p-1][1], color='red', marker = 'o', linestyle = 'none')
plt.show()
