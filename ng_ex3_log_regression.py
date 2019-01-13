# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 12:18:43 2018

@author: Michael
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from random import randint

# load MATLAB files
from scipy.io import loadmat
from scipy.optimize import minimize

from sklearn.linear_model import LogisticRegression

data = loadmat('data/ex3/ex3data1.mat')
data.keys()
x=data['X']
y=data['y']


def displayData(data,nrows,ncols):

    x_all = np.array([np.zeros(ncols*20)])
    rowsint = [randint(0,500) for l in range(1,ncols+1)]
    for i in rowsint:
        xr=np.reshape(x[(i-1)*ncols],(20,20))
        for j in range(1,ncols):
            xr_temp=np.reshape(x[(i-1)*ncols+j],(20,20))
            xr=np.concatenate((xr,xr_temp),axis=1)
        x_all = np.concatenate((x_all, xr),axis=0)
    plt.imshow(x_all.T, cmap="gray")
    plt.show()
    return [1]

displayData(x,10,10)

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


def oneVsAll(x,y,numLabels,lam):
    theta_all=[]
    theta_size = x[0].shape[0]
    for k in range(1,numLabels+1):
        theta_start = np.zeros(theta_size)
        y_train = np.array([1 if (y[x_i-1] == k) else 0 for x_i in range(1,ss_train+1)])
        result = gradientDesc_reg(50000,0.0000001,theta_start,x,y_train,0.1,lam)
        print("Finished training set: ", k)
        print("Cost funtion: " + str(result[0][0]))
        print("Iterations: " + str(result[2]))
        theta_all.append(result[1])
    return np.array(theta_all)

def predictOneVsAll(x,y,theta):
    prob = x.dot(theta.T)
    guesses = []
    count =0
    correctCount = 0
    for p in prob:
        guess = p.argmax()+1
        guesses.append(guess)
        if(guess == y[count]): correctCount += 1
        count +=1
    return [guesses,correctCount/y.shape[0]]
    

# TRAIN
numLabels=10
k=1
lam =0.2
y=np.concatenate(y)
ss_train = y.shape[0]
x = np.hstack([np.ones((ss_train,1)),x])
theta = oneVsAll(x,y,numLabels,lam)

plt.imshow(np.reshape(theta[3][1:],(20,20)).T, cmap="gray")

# Test
prediction = predictOneVsAll(x,y,theta)
