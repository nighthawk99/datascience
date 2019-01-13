# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt #sets up plotting under plt
import random
from scipy.io import loadmat
from sklearn import svm
import math
import scipy.optimize as op

data = loadmat('data/ex6/ex6data1.mat')
data.keys()
x=data['X']
y=data['y']

def plot2DXY(x,y):
    for p in range(1,len(y)+1):
      if y[p-1]==0:
        plt.plot(x[p-1][0],x[p-1][1], color='green', marker = 'o', linestyle = 'none') 
      else:
        plt.plot(x[p-1][0],x[p-1][1], color='red', marker = 'o', linestyle = 'none')
    plt.xlim(np.min(x),np.max(x))
    plt.show()

def plot2DXY_withline(x,y,model):
    for p in range(1,len(y)+1):
      if y[p-1]==0:
        plt.plot(x[p-1][0],x[p-1][1], color='green', marker = 'o', linestyle = 'none') 
      else:
        plt.plot(x[p-1][0],x[p-1][1], color='red', marker = 'o', linestyle = 'none')
    x_vals = np.linspace(np.min(x),np.max(x),100)
    y_vals = (-x_vals*svm_lin.coef_[0][0]-svm_lin.intercept_[0])/svm_lin.coef_[0][1]
    plt.plot(x_vals,y_vals)
    plt.xlim(np.min(x),np.max(x))
    plt.show()

    
#def SVM(x,y,C, kern):
kern='linear'
C=1000000

y = y.flatten()
svm_lin = svm.SVC(C,kernel=kern)
svm_lin.fit(x,y)
plot2DXY_withline(x,y,svm_lin)
