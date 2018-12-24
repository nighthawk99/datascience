# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 12:43:21 2018

@author: micha
"""

#%matplotlib inline
import numpy as np # imports a fast numerical programming library
import scipy as sp #imports stats functions, amongst other things
import matplotlib as mpl # this actually imports matplotlib
import matplotlib.cm as cm #allows us easy access to colormaps
import matplotlib.pyplot as plt #sets up plotting under plt
import pandas as pd #lets us handle data as dataframes
import math

#sets up pandas table display
#pd.set_option('display.width', 500)
#pd.set_option('display.max_columns', 100)
#pd.set_option('display.notebook_repr_html', True)

import seaborn as sns #sets up styles and gives us more plotting options


def sigmoid(x):
    return 1/(1+np.exp(-x))

def costFunction(theta,x,y):
    ssize = df_x.shape[0]
    size_theta = theta.shape[0]
    cF = (-y.dot(np.log(sigmoid(x.dot(theta)))) - (1-y).dot(np.log(1-sigmoid(x.dot(theta))))) /ssize
    x_split = np.hsplit(x, size_theta)
    x_split = np.array([np.reshape(x_i,ssize) for x_i in x_split])
    grad = x_split.dot((sigmoid(x.dot(theta)) - y)) /ssize
    return [cF, grad]

def gradientDesc(maxIter, precision, init_theta, x,y,stepmult):
    stepSizes =[]
    costF=0
    prev_stepSize =1
    costF_list = []
    theta_list=[]
    iters = 0
    theta = init_theta
    costF = costFunction(theta,x,y)
    costF_list.append(costF[0])
    theta_list.append(theta)

    while prev_stepSize > precision and iters < maxIter:
        val=costF[0]
        grad = costF[1]
        theta = theta - stepmult * grad
        costF = costFunction(theta,x,y)
        prev_stepSize = abs(costF[0]-val)
        stepSizes.append(prev_stepSize)
        costF_list.append(costF[0])
        theta_list.append(theta)
        iters=iters + 1
    return [costF, theta,iters,stepSizes, costF_list, theta_list]


# Exercise 1

df_x = pd.read_csv("data/ex2/ex2data1.csv", sep=",", usecols = [0,1], header=None, engine='python')
df_y = pd.read_csv('data/ex2/ex2data1.csv', sep=",", usecols = [2], header=None, engine='python')
df_y = df_y.astype(int)
df_x.head()

input_x_scale =100.0
x = df_x.values
x = x = x/input_x_scale
x = np.hstack([np.ones((df_x.shape[0], 1)), x]) #values return numpy version of panda
y = df_y[df_y.columns.values[0]].values


theta = np.array([0.0,0.0,0.0])
gD=gradientDesc(1000000000,0.00000000000001,theta, x, y,0.1)

theta = gD[1]

print("Cost funtion: " + str(gD[0][0]))
print("Iterations: " + str(gD[2]))
print("Theta: " + str(gD[1]))

plt.plot(gD[3][2:], color='green', marker ='o')
plt.ylim(0,0.00001)
plt.show()

y_intercept = (0.5-theta[0])/theta[2]
slope = -theta[1]/theta[2]
decBoundary = [slope*k/input_x_scale+y_intercept for k in range(1,100)]
decBoundary_x = [m/input_x_scale for m in range(1,100)]

plt.plot(decBoundary_x,decBoundary)
for p in range(1,len(y)):
  if y[p-1]==0:
    plt.plot(x[p-1][1],x[p-1][2], color='green', marker = 'o', linestyle = 'none') 
  else:
    plt.plot(x[p-1][1],x[p-1][2], color='red', marker = 'o', linestyle = 'none')

plt.xlabel("x")
plt.ylabel("y")
plt.xlim(0.3,1)
plt.ylim(0.3,1)
plt.show()

test = np.array([1,0.45,0.85])
sigmoid(test.dot(theta))