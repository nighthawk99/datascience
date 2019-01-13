# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt #sets up plotting under plt
import random
from scipy.io import loadmat
import math
import scipy.optimize as op

data = loadmat('data/ex3/ex3data1.mat')
data.keys()
x=data['X']
y=data['y']

data = loadmat('data/ex3/ex3weights.mat')
data.keys()
weight_set1 = data['Theta1']
weight_set2 = data['Theta2']
NN = list([weight_set1, weight_set2])

def displayData(data,nrows,ncols):

    x_all = np.array([np.zeros(ncols*20)])
    rowsint = [random.randint(0,500) for l in range(1,ncols+1)]
    for i in rowsint:
        xr=np.reshape(x[(i-1)*ncols],(20,20))
        for j in range(1,ncols):
            xr_temp=np.reshape(x[(i-1)*ncols+j],(20,20))
            xr=np.concatenate((xr,xr_temp),axis=1)
        x_all = np.concatenate((x_all, xr),axis=0)
    plt.imshow(x_all.T, cmap="gray")
    plt.show()
    return [0]

def displayHiddenUnits(theta,ncols=5):
    x_all = np.array([np.zeros(ncols*20+1)])
    xr = np.reshape(np.array([np.zeros(20)]),(20,1))
    colCount =0
    for unit in optTheta[0]:
        xr_temp=np.reshape(unit[1:],(20,20))
        if(colCount < ncols):
            print(colCount)
            xr=np.concatenate((xr,xr_temp),axis=1)
            colCount +=1
            #plt.imshow(xr.T, cmap="gray")
            #plt.show()
        else:    
            x_all = np.concatenate((x_all, xr),axis=0)
            xr = np.reshape(np.array([np.zeros(20)]),(20,1))
            xr=np.concatenate((xr,xr_temp),axis=1)
            colCount =1        
    x_all = np.concatenate((x_all, xr),axis=0)
    plt.imshow(x_all, cmap="gray")
    plt.show()
    return [1]

def displaySelect(x,which):
    for i in which:
        xr=np.reshape(x[i],(20,20))
        plt.imshow(xr.T, cmap="gray")
        plt.show()
    return 0

def sigmoid(x):
    return 1/(1+np.exp(-x))

def gradsigmoid(x):
    g = sigmoid(x)
    return g*(1-g)

def chooseEps(in_layer_size,out_layer_size):
    return math.sqrt(6)/(math.sqrt(in_layer_size + out_layer_size))

def generateNeuralNetwork(layerSizes, eps): #layersizes without bias unit
    network = []
    layers = len(layerSizes)
    for lay in range(1,layers):
        network.append(np.array([[2*(random.random()-0.5)*eps for _ in range(layerSizes[lay-1]+1)] for _ in range(layerSizes[lay])]))
    return network    
        
def feedForward(x,neural_network): #x has no bias term, neural_network has bias term
    activations_list = []
    activation = x
    for layer in neural_network:
        activation_size = activation.shape[0]
        activation = np.hstack([np.ones((activation_size,1)),activation])
        activation_input = activation.dot(layer.T)
        activation = sigmoid(activation_input)
        activations_list.append(np.array([activation_input,activation]))
    return activations_list

def makePrediction(x,y, neural_network):
    activations_list = feedForward(x, neural_network)
    output = activations_list[-1][1]
    guesses = []
    count =0
    correctCount = 0
    for out in output:
        guess = out.argmax()+1
        if(guess == y[count][0]): correctCount += 1
        guesses.append(guess)
        count +=1
    acc = correctCount/y.shape[0]
    print("Accuracy: %f." %(acc))
    return [guesses,acc]

def vectorize_labels(y,numLabels):
    y_labels = np.array([[0 for _ in range(1,numLabels+1)] for _ in range(1,y.shape[0]+1)])
    m = y.shape[0]
    for i in range(1,m+1): y_labels[i-1][y[i-1][0]-1] = 1
    return y_labels
   
def nnCostFunction(theta,x,y_vect, sampleSize):
    numLabels = theta[-1].shape[0]
    numLayers = len(theta)+1
    y_guesses = feedForward(x,theta)[numLayers-2][1]
    one_matrix = np.ones((sampleSize,numLabels))
    one_vector = np.ones((numLabels))
    cF = -np.sum(([yi.dot(xi) for (yi,xi) in zip(y_vect,np.log(y_guesses))] +[yi.dot(xi) for (yi,xi) in zip((one_matrix-y_vect),np.log(one_vector-y_guesses))]))/sampleSize
    return cF

def nnCostFunction_reg(theta,x,y_vect,sampleSize,lam):
    numLabels = theta[-1].shape[0]
    numLayers = len(theta)+1
    y_guesses = feedForward(x,theta)[numLayers-2][1]
    one_matrix = np.ones((sampleSize,numLabels))
    one_vector = np.ones((numLabels))
    cF = -np.sum(([yi.dot(xi) for (yi,xi) in zip(y_vect,np.log(y_guesses))] +[yi.dot(xi) for (yi,xi) in zip((one_matrix-y_vect),np.log(one_vector-y_guesses))]))/sampleSize
    dimLayer0 = theta[0][0].shape[0]-1 #with bias unit
    dimLayer1 = theta[0].shape[0]
    dimLayer2 = theta[1].shape[0]
    reg = lam/(2*sampleSize) * (sum([theta[0][i][j]*theta[0][i][j] for i in range(0,dimLayer1) for j in range(1,dimLayer0)])
        +sum([theta[1][i][j]*theta[1][i][j] for i in range(0,dimLayer2) for j in range(1,dimLayer1+1)]))
    return cF+reg

#unrolled version required by optimizer
def nnCostFunction_reg_unrolled(unrolledtheta,x_unrolled,y_unrolled,sampleSize,lam):
    theta = rollData(unrolledtheta)
    x = rollDataX(x_unrolled)
    y_vect = vectorize_labels(y_unrolled,10)
    numLabels = theta[-1].shape[0]
    numLayers = len(theta)+1
    y_guesses = feedForward(x,theta)[numLayers-2][1]
    one_matrix = np.ones((sampleSize,numLabels))
    one_vector = np.ones((numLabels))
    cF = -np.sum(([yi.dot(xi) for (yi,xi) in zip(y_vect,np.log(y_guesses))] +[yi.dot(xi) for (yi,xi) in zip((one_matrix-y_vect),np.log(one_vector-y_guesses))]))/sampleSize
    dimLayer0 = theta[0][0].shape[0]-1 #with bias unit
    dimLayer1 = theta[0].shape[0]
    dimLayer2 = theta[1].shape[0]
    reg = lam/(2*sampleSize) * (sum([theta[0][i][j]*theta[0][i][j] for i in range(0,dimLayer1) for j in range(1,dimLayer0)])
        +sum([theta[1][i][j]*theta[1][i][j] for i in range(0,dimLayer2) for j in range(1,dimLayer1+1)]))
    return cF+reg

def backPropagate(theta,x,y_vect,sampleSize, lam):
    Delta2 = 0
    Delta1 = 0
    for t in range(0,sampleSize):
        # Step 1
        activations_list=feedForward(np.array([x[t]]),theta)
        activations_list_with_bias = np.hstack([np.ones((1,1)),activations_list[0][1]])
        # Step 2
        delta_3 = (activations_list[1][1]-y_vect[t]) #(10,1) #ativations do not contain bias, esp. not in output layer
        # Step 3
        delta_2 = ((theta[1].T)[1:]).dot(delta_3[0].T) * gradsigmoid(activations_list[0][0]) #adj. theta is (25,10)
                                                            #, delta_3 (10,1),activations_list (25,1), -> (25,1)
        # Step 4
        Delta2 += (delta_3.T).dot(activations_list_with_bias) # (10,1) (1,26), -> (10,26)
        Delta1 += (delta_2.T).dot(np.array([np.hstack([np.ones(1),x[t]])])) #(25,1), (1,401) -> (25,401)
    
    Delta2 = Delta2/sampleSize
    Delta1 = Delta1/sampleSize
    
    # Regularization
    
    Delta1[:][1:] = Delta1[:][1:] + float(lam)/sampleSize * theta[0][:][1:]
    Delta2[:][1:] = Delta2[:][1:] + float(lam)/sampleSize * theta[1][:][1:]
    
    return [Delta1, Delta2]

#unrolled version required by optimizer
def backPropagate_unrolled(unrolledtheta,x_unrolled,y_unrolled,sampleSize, lam):
    Delta2 = 0
    Delta1 = 0
    theta = rollData(unrolledtheta)
    x = rollDataX(x_unrolled)
    y_vect = vectorize_labels(y_unrolled,10)
    for t in range(0,sampleSize):
        # Step 1
        activations_list=feedForward(np.array([x[t]]),theta)
        activations_list_with_bias = np.hstack([np.ones((1,1)),activations_list[0][1]])
        # Step 2
        delta_3 = (activations_list[1][1]-y_vect[t]) #(10,1) #ativations do not contain bias, esp. not in output layer
        # Step 3
        delta_2 = ((theta[1].T)[1:]).dot(delta_3[0].T) * gradsigmoid(activations_list[0][0]) #adj. theta is (25,10)
                                                            #, delta_3 (10,1),activations_list (25,1), -> (25,1)
        # Step 4
        Delta2 += (delta_3.T).dot(activations_list_with_bias) # (10,1) (1,26), -> (10,26)
        Delta1 += (delta_2.T).dot(np.array([np.hstack([np.ones(1),x[t]])])) #(25,1), (1,401) -> (25,401)
    
    Delta2 = Delta2/sampleSize
    Delta1 = Delta1/sampleSize
    
    # Regularization
    
    Delta1[:][1:] = Delta1[:][1:] + float(lam)/sampleSize * theta[0][:][1:]
    Delta2[:][1:] = Delta2[:][1:] + float(lam)/sampleSize * theta[1][:][1:]
    
    return unrollData([Delta1, Delta2])
    
def unrollData(l):
    unrolled = []
    for item in l:
        unrolled.append(np.reshape(item, item.shape[0]*item[0].shape[0]))
    return np.hstack(np.array(unrolled))

def rollData(unrolled):
    item1 = np.reshape(unrolled[:(25*401)],(25,401))
    item2 = np.reshape(unrolled[(25*401):],(10,26))
    return [item1,item2]

def rollDataX(x):
    return np.reshape(x,(5000,400))

def unrollDataX(x):
    return np.reshape(x,(1,x.shape[0]*x[0].shape[0]))
    
def checkGradient(theta,Delta,x,y_vect,sample_size,lam): #25x401 10x26
    unrolledTheta = unrollData(theta)
    unrolledDelta = unrollData(Delta)
    length = len(unrolledTheta)
    picks = [int(np.random.rand()*length) for i in range(1,6)]
    myeps = 0.0001
    for i in picks:
        theta_eps = np.zeros(length)
        theta_eps[i] = myeps
        theta_p=rollData(unrolledTheta + theta_eps)
        theta_n=rollData(unrolledTheta - theta_eps)
        cost_p = nnCostFunction_reg(theta_p,x,y_vect, sample_size,lam)
        cost_n = nnCostFunction_reg(theta_n,x,y_vect, sample_size,lam)
        numGrad = (cost_p-cost_n)/(2*float(myeps))
        print("For pick %d. we have Delta: %f. vs Delta numerical %f." %(i, unrolledDelta[i],numGrad))


# Test ForwardProp

theta = generateNeuralNetwork(layerSizes = [400,25,10], eps = chooseEps(400,10))
y_vect = vectorize_labels(y, numLabels = theta[1].shape[0])
lam = 1
sampleSize = y.shape[0]

cF = nnCostFunction_reg(theta,x,y_vect, sampleSize,lam)
print(cF)

# Test Gradient
Delta = backPropagate(theta,x,y_vect,sampleSize, lam)
checkGradient(theta,Delta,x,y_vect, sampleSize,lam)

# Run Optimizer
theta_unrolled = unrollData(theta)
x_unrolled = unrollDataX(x)

opt = op.fmin_cg(f = nnCostFunction_reg_unrolled, x0 = theta_unrolled, fprime=backPropagate_unrolled, args = (x_unrolled,y,sampleSize,lam), maxiter=50,disp=1,full_output=1)
optTheta = rollData(opt[0])
pred = makePrediction(x,y,optTheta)
displayHiddenUnits(optTheta,5)



