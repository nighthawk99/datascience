# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import csv
import math
import pandas as pd

# Prepare data

with open('ex2/ex2data1.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)
    

x,y,z = zip(*your_list)
x = [float(x_i) for x_i in x]
y = [float(y_i) for y_i in y]
z = [float(z_i) for z_i in z]
    
features=[]
for k in range(0, len(x)):
  features.append(1) #intercept
  features.append(x[k])
  features.append(y[k])
  
# Plot point grapgh
  
for p in range(1,len(x)):
  if z[p-1]==0:
    plt.plot(x[p-1],y[p-1], color='green', marker = 'o', linestyle = 'none') 
  else:
    plt.plot(x[p-1],y[p-1], color='red', marker = 'o', linestyle = 'none')
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# define log regression

def vector_subtract(x,y):
    return [x_i-y_i for x_i,y_i in zip(x,y)]

def dotproduct(x,y):
  return sum([x_i*y_i for x_i,y_i in zip(x,y)])

def scalar_multiply(alpha,x):
    return [alpha * x_i for x_i in x]

def matrix_vector_product(M,x):
  if not(isinstance(M,list)):
    M=[M]
    x = [x]
  lenX = len(x)
  lenM = len(M)
  out =[]
  for i in range(0,int(lenM/lenX)):
    out.append(dotproduct(M[(i*lenX):(lenX+i*lenX)],x))
  return out


def sigmoid(delta, x):
  z = matrix_vector_product(delta,x)
  return [1/(1+math.exp(-z_i)) for z_i in z]


def costFunction(theta,X,Y):
  sampleSize = len(Y)
  featureLength = len(theta)
  grad =[0]*featureLength
  J=0
  for i in range(0,sampleSize):
    h = sigmoid(X[(i*featureLength):(i*featureLength+(featureLength))],theta)[0]
    J = J + (-Y[i]*math.log(h)-(1-Y[i])*math.log(1-h))
    diff = h - Y[i]
    for j in range(0, featureLength):
      grad[j]=grad[j]+(diff*features[(i*featureLength)+j])/sampleSize
  J = J/sampleSize
  return [J,grad]


def gradientDesc(maxIter, precision, init_theta, X,Y,stepmult):
    stepSizes =[]
    costF=0
    prev_stepSize =1
    costF_list = []
    theta_list=[]
    iters = 0
    theta = init_theta
    costF = costFunction(theta,X,Y)
    costF_list.append(costF[0])
    theta_list.append(theta)

    while prev_stepSize > precision and iters < maxIter:
        val=costF[0]
        theta = vector_subtract(theta,scalar_multiply(stepmult,costF[1]))
        costF = costFunction(theta,X,Y)
        prev_stepSize = abs(costF[0]-val)
        stepSizes.append(prev_stepSize)
        costF_list.append(costF[0])
        theta_list.append(theta)
        iters=iters + 1
    return [costF, theta,iters,stepSizes, costF_list, theta_list]
    

theta = [0.0,0.0,0.0]
gD=gradientDesc(250,0.01,theta, features, z,0.001)

costF = gD[0][0]
theta = gD[1]
iters = gD[2]

print("Cost funtion: " + str(costF))
print("Iterations: " + str(iters))
print("Theta: " + str(theta))

plt.plot(gD[3][2:], color='green', marker ='o')
plt.show()

y_intercept = (0.5-theta[0])/theta[2]
slope = -theta[1]/theta[2]
decBoundary = [slope*k+y_intercept for k in range(1,100)]

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

test = [1,45,85]
sigmoid(test,theta)

#Test

#x0 +x1*a +x2*b = 0.5
#b = -x1/x2*a + (0.5-x0)/x2

#Cost funtion: 0.213111560994
#Iterations: 1668860
#Theta: [-18.141182972810896, 0.1501504660518356, 0.14465601805796394]
#my_theta= [-18.141182972810896, 0.1501504660518356, 0.14465601805796394]
#myCF= costFunction(my_theta,features,z)

