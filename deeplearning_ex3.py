# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #sets up plotting under plt
import random
from scipy.io import loadmat

def sigmoid(x):
    return 1/(1+np.exp(-x))

def generateNeuralNetwork(layersizes): #layersizes without bias unit
    network = []
    layers = len(layersizes)
    for lay in range(1,layers):
        network.append(np.array([[random.random() for _ in range(layersizes[lay-1]+1)] for _ in range(layersizes[lay])]))
    return network    
        
def feedForward(x,neural_network): #x has no bias term, neural_network has bias term
    activations_list = []
    activation = x
    for layer in neural_network:
        activation_size = activation.shape[0]
        activation = np.hstack([np.ones((activation_size,1)),activation])
        activation = sigmoid(activation.dot(layer.T))
        activations_list.append(activation)
        print(1)
    return activations_list    

def makePrediction(x,y, neural_network):
    activations_list = feedForward(x, neural_network)
    output = activations_list[len(activations_list)-1]
    guesses = []
    count =0
    correctCount = 0
    for out in output:
        guess = out.argmax()+1
        if(guess == y[count]): correctCount += 1
        guesses.append(guess)
        count +=1
    return [guesses,correctCount/y.shape[0]]
        
def showPredictions(x,guesses):
    i=0
    for xr in x:
        xr=np.reshape(xr,(20,20))
        plt.imshow(xr.T, cmap="gray")
        plt.show()
        print("Guess: ",y[i])
        i+=1
    return 0

def showSelect(x,guesses,which):
    i=0
    for i in which:
        xr=np.reshape(x[i],(20,20))
        plt.imshow(xr.T, cmap="gray")
        plt.show()
        print("Guess: ",guesses[i])
        print("Actual: ",y[i])
    return 0

def findErrorPredictions(guesses,y):
    i=0
    wrong=[]
    for guess in guesses:
        if(y[i]!=guess):
            wrong.append(i)
        i+=1
    return wrong    
    
    
data = loadmat('data/ex3/ex3data1.mat')
data.keys()
x=data['X']
y=data['y']

data = loadmat('data/ex3/ex3weights.mat')
data.keys()
weight_set1 = data['Theta1']
weight_set2 = data['Theta2']
neural_network = list([weight_set1, weight_set2])

test = makePrediction(x,y,neural_network)
showPredictions(x,test[0])
wrongs = findErrorPredictions(test[0],y)
showSelect(x,test[0],wrongs)


#activations_list = []
#activation = np.array([[1,2,3],[5,6,7]])
#NN = generateNeuralNetwork([400,25,10])
#activations = feedForward(x,NN)

