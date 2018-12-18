# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 12:43:21 2018

@author: micha
"""

%matplotlib inline
import numpy as np # imports a fast numerical programming library
import scipy as sp #imports stats functions, amongst other things
import matplotlib as mpl # this actually imports matplotlib
import matplotlib.cm as cm #allows us easy access to colormaps
import matplotlib.pyplot as plt #sets up plotting under plt
import pandas as pd #lets us handle data as dataframes

#sets up pandas table display
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)

import seaborn as sns #sets up styles and gives us more plotting options

df_x = pd.read_csv("data/ex2/ex2data1.csv", sep=",", usecols = [0,1], header=None, engine='python')
df_y = pd.read_csv('data/ex2/ex2data1.csv', sep=",", usecols = [2], header=None, engine='python')
df_y = df_y.astype(int)
df_x.head()

x = np.hstack([np.ones((df_x.shape[0], 1)), df_x.values]) #values return numpy version of panda
y = df_y["y"].values


def sigmoid(x):
    return 1/(1+np.exp(-x))

def grad_l(theta, x, y):
    z = y*x.dot(theta)
    g = -np.mean((1-sigmoid(z))*y*x.T, axis=1)
    return g

def hess_l(theta, x, y):
    hess = np.zeros((x.shape[1], x.shape[1]))
    z = y*x.dot(theta)
    for i in range(hess.shape[0]):
        for j in range(hess.shape[0]):
            if i <= j:
                hess[i][j] = np.mean(sigmoid(z)*(1-sigmoid(z))*x[:,i]*x[:,j])
                if i != j:
                    hess[j][i] = hess[i][j] 
    return hess