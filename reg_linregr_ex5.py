# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt #sets up plotting under plt
import random
from scipy.io import loadmat
import math
import scipy.optimize as op


def plotXY_2D(theta, x_plot, x,y):
    plt.plot(x_plot,y,color='green', marker = 'o', linestyle = 'none')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x_plot,x.dot(theta),color='red',marker = 'x', linestyle = 'none')
    #plt.xlim(np.min(x),np.max(x))
    #plt.ylim(np.min(y),np.max(y))
    plt.show()
    

def linRegCostFunction_reg(theta,x,y,lam):    #x does not contain bias term
    #x = np.hstack((np.ones((x.shape[0],1)),x))
    m = x.shape[0]
    theta = theta.reshape((theta.shape[0],1))
    mult = np.dot(x,theta).reshape((m,1))-y
    cf = float((1.0/(2*m)) * (np.dot(mult.T,mult)))
    reg = float((lam/(2*m)) * (np.dot(theta[1:].T,theta[1:])))
    return cf+reg


def linRegGradient_reg(theta,x,y,lam):    #x does not contain bias term
    #x = np.hstack((np.ones((x.shape[0],1)),x))
    theta = theta.reshape((theta.shape[0],1))
    m = x.shape[0]
    reg = lam * theta
    reg[0] = 0.0
    mult = x.dot(theta).reshape((m,1))-y
    grad = float(1./float(m)) * ((x.T).dot(mult) + reg)
    return grad

def linRegGradient_reg_flat(theta,x,y,lam):
    return linRegGradient_reg(theta,x,y,lam).flatten()

##

def optimizeTheta(theta, x, y, lam=0.,print_output=True):
    opt = op.fmin_cg(f = linRegCostFunction_reg, x0 = theta, fprime=linRegGradient_reg_flat, args = (x,y,lam), maxiter=4000,disp=1,full_output=1)
    opt = opt[0].reshape((theta.shape[0],1))
    return opt

####
    
def learningCurve(theta,x_train,y_train, x_val, y_val,lam):
    size =[]
    error_train=[]
    error_val=[]
    for i in range(2,x_train.shape[0]):
        opt_theta = optimizeTheta(theta, x_train[1:i,], y_train[1:i], lam,print_output=True)        
        size.append(i)
        error_train.append(linRegCostFunction_reg(opt_theta,x_train[1:i,],y_train[1:i],0.))
        error_val.append(linRegCostFunction_reg(opt_theta,x_val,y_val,0.))
    plt.plot(size,error_train,color='red')
    plt.plot(size,error_val,color='blue')
    plt.show()
    return [size,error_train, error_val]

def transformPoly(x,k):
    x_new = np.array([[math.pow(x_i,1)] for x_i in x]).reshape((x.shape[0],1))
    for w in range(2,k+1):
        x_new = np.hstack((x_new, np.array([[math.pow(x_i,w)] for x_i in x]).reshape((x.shape[0],1))))
    return x_new      

def featureNormalization(x): #x without bias term
    means = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x-means)/std

def validationCurve(theta, x_train, y_train,x_val,y_val, lam_arr):
    error_train=[]
    error_val=[]
    for lam in lam_arr:
        opt_theta = optimizeTheta(theta, x_train, y_train, lam,print_output=True)        
        error_train.append(linRegCostFunction_reg(opt_theta,x_train,y_train,0.))
        error_val.append(linRegCostFunction_reg(opt_theta,x_val,y_val,0.))
    plt.plot(lam_arr,error_train,color='red')
    plt.plot(lam_arr,error_val,color='blue')
    plt.show()
        
        
# Load Input      

data = loadmat('data/ex5/ex5data1.mat')
data.keys()
x=data['X']
y=data['y']

xval = data['Xval']
yval = data['yval']

xtest = data['Xtest']
ytest = data['ytest']

x = np.hstack((np.ones((x.shape[0],1)),x))
xval = np.hstack((np.ones((xval.shape[0],1)),xval))
xtest = np.hstack((np.ones((xtest.shape[0],1)),xtest))

theta = np.array([[1],[1]])
lam=1.0

#Test
cf=linRegCostFunction_reg(theta,x,y,lam)
grad=linRegGradient_reg(theta,x,y,lam)
print(cf)
print(grad)

opt = optimizeTheta(theta, x, y, lam,print_output=True)
plotXY_2D(opt,data['X'],x,y)

#Print Learning Curves

learningcurves = learningCurve(theta,x,y, xval, yval,lam)

#Poly test
lam=1.0
x=data['X']
x_poly_norm = featureNormalization(transformPoly(x,8))
x_poly_norm = np.hstack((np.ones((x_poly_norm.shape[0],1)),x_poly_norm))
theta = np.array([[1] for i in range(1,10)])
opt_theta = optimizeTheta(theta,x_poly_norm,y,lam=0)
plotXY_2D(opt_theta,data['X'],x_poly_norm,y)

# Lam Validation
lam_arr = np.array([0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10])

x=data['X']
x_poly_norm = featureNormalization(transformPoly(x,8))
x_poly_norm = np.hstack((np.ones((x_poly_norm.shape[0],1)),x_poly_norm))
theta = np.array([[1] for i in range(1,10)])

xval = data['Xval']
yval = data['yval']

x_val_poly_norm = featureNormalization(transformPoly(xval,8))
x_val_poly_norm = np.hstack((np.ones((x_val_poly_norm.shape[0],1)),x_val_poly_norm))
validationCurve(theta, x_poly_norm, y,x_val_poly_norm,yval, lam_arr)

