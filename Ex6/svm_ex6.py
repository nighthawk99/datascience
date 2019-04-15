# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt #sets up plotting under plt
import random
from scipy.io import loadmat
from sklearn import svm
import math
import scipy.optimize as op

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
    
def plotDecisionBoundary(x,y, model):
    for p in range(1,len(y)+1):
      if y[p-1]==0:
        plt.plot(x[p-1][0],x[p-1][1], color='green', marker = 'o', linestyle = 'none') 
      else:
        plt.plot(x[p-1][0],x[p-1][1], color='red', marker = 'o', linestyle = 'none')
 
    h = .02  # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    #plt.figure(figsize=(6,3))
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
    plt.xlim(np.min(x),np.max(x))
    plt.ylim(np.min(y), np.max(y)) 
    plt.show()
    
def gaussian(x1,x2,sigma):
    return math.exp(-math.pow(np.linalg.norm(x1-x2),2) / (2*math.pow(sigma,2)))

def predictionError(xval,yval,model):
        results = model.predict(xval)

# LINEAR Kernel  
    
data = loadmat('data/ex6/ex6data1.mat')
data.keys()
x=data['X']
y=data['y']
    
kern='linear'
y = y.flatten()
C=1
#Note that sklearn svm adds the intercept coefficient by itself
svm_lin = svm.SVC(C,kernel=kern)
svm_lin.fit(x,y)
plot2DXY_withline(x,y,svm_lin)

# Gaussian Kernel   

data = loadmat('data/ex6/ex6data2.mat')
data.keys()
x=data['X']
y=data['y']

kern='rbf'
y = y.flatten()
C=1000
sigma = 0.5
#Note that sklearn svm adds the intercept coefficient by itself
svm_lin = svm.SVC(C,kernel=kern,gamma=math.pow(sigma,-2))
svm_lin.fit(x,y)
plotDecisionBoundary(x,y,svm_lin)


# Testing different C and sigma

data = loadmat('data/ex6/ex6data3.mat')
data.keys()
x=data['X']
y=data['y']
xval=data['Xval']
yval=data['yval']

Cs = np.array([0.01,0.03,0.1,0.3,1,3,10,30]) 
sigmas = np.array([0.01,0.03,0.1,0.3,1,3,10,30])
combined = [(C,sigma) for C in Cs for sigma in sigmas]
accuracy_curr = 0.0
comb_curr = [-1,-1]

for pair in combined:
    svm_lin = svm.SVC(pair[0],kernel=kern,gamma=math.pow(pair[1],-2))
    svm_lin.fit(x,y)
    accuracy = float(sum(svm_lin.predict(xval)==yval.ravel()))/float(len(xval))
    if accuracy > accuracy_curr:
        accuracy_curr = accuracy
        comb_curr = pair

print("Best accuracy: ", accuracy_curr)
print("Obtained with: [", comb_curr[0],comb_curr[1],"]")
svm_lin = svm.SVC(comb_curr[0],kernel=kern,gamma=math.pow(comb_curr[1],-2))
svm_lin.fit(x,y)
plotDecisionBoundary(x,y,svm_lin)
plotDecisionBoundary(xval,yval,svm_lin)
        
    

