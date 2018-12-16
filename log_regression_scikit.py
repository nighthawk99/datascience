# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 14:01:57 2018

@author: micha
"""

import matplotlib.pyplot as plt
import csv
import math
import pandas as pd

from sklearn.linear_model import LogisticRegression

with open('data/ex2/ex2data1.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)
    
x,y,z = zip(*your_list)
x = [float(x_i) for x_i in x]
y = [float(y_i) for y_i in y]
z = [float(z_i) for z_i in z]
     
featuresWI=[]
for k in range(0, len(x)):
  featuresWI.append([x[k],y[k]])

clf = LogisticRegression(random_state = 0, solver='newton-cg',fit_intercept = True).fit(featuresWI,z)
clf.predict_proba([[45,85]])
clf.score(featuresWI,z)
df_coef = clf.coef_
df_interc = clf.intercept_

y_intercept = (0.5-df_interc)/df_coef[0][1]
slope = -df_coef[0][0]/df_coef[0][1]
decBoundary = [slope * k +y_intercept for k in range(1,100)]

plt.plot(range(1,100),decBoundary)
for p in range(1,len(x)):
  if z[p-1]==0:
    plt.plot(x[p-1],y[p-1], color='green', marker = 'o', linestyle = 'none') 
  else:
    plt.plot(x[p-1],y[p-1], color='red', marker = 'o', linestyle = 'none')
    
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(30,100)
plt.ylim(30,100)
plt.show()

## Run test against own implementation

scikit_theta = [df_interc, df_coef[0][0],df_coef[0][1]]
my_theta= [-18.141182972810896, 0.1501504660518356, 0.14465601805796394]

#myCF= costFunction(my_theta,features,z)
#scikitCF= costFunction(scikit_theta,features,z)