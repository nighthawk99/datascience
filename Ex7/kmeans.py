import numpy as np
import pandas as pd
import random as rd
from scipy.io import loadmat
import matplotlib.pyplot as plt #sets up plotting under plt
import cv2

def kMeansInitCentroids(X,k):
    lenX = len(X)
    centroidargs = [int(rd.random()*lenX) for _ in range(0,k)]
    return X[centroidargs]

def kMeansAdvInitCentroids(X,k):
    lenX = len(X)
    centroidargs = np.array([])
    while len(centroidargs)<k:
        add_index = int(rd.random()*lenX)
        if not(add_index in centroidargs): centroidargs = np.append(centroidargs, add_index)
    return X[centroidargs.astype(int)]


def distance(x,y):
    return sum((float(x_i)-float(y_i))*(float(x_i)-float(y_i)) for x_i,y_i in zip(x,y))


def findClosestCentroid(x, centroids):
    idx=0
    dist = distance(x,centroids[0])
    for j in range(1, len(centroids)):
        if(dist > distance(x,centroids[j])):
            dist = distance(x,centroids[j])
            idx = j
    return int(idx)

def findClosestCentroids(X, centroids):
    idx = np.array([])
    for x in X:
        idx=np.append(idx,findClosestCentroid(x,centroids))
    return idx

def findNewCentroids(X,idx,kMax):
    newCentroids=np.array([])
    for k in range(0,kMax):
        centroidargs = np.where(idx == k)
        if(len(centroidargs) > 0):
            newCentroids=np.append(newCentroids,np.array(np.mean(X[centroidargs],axis =0)))
    return np.split(newCentroids,len(newCentroids)/len(X[0]),0)
        

def plotCentroids(data,centroids):
    plt.plot([data[i][0] for i in range(0,len(data))],[data[i][1] for i in range(0,len(data))],
              color='yellow', marker = 'o', linestyle = 'none')
    plt.plot([centroids[i][0] for i in range(0,len(centroids))],[centroids[i][1] for i in range(0,len(centroids))],
              color='blue', marker = '^', linestyle = 'none')
    plt.show()
    
def costF(X,idx,centroids):
    dist = 0
    for j in range(0,len(idx)):
        dist = dist + distance(X[j],centroids[int(idx[j])])
    return dist


def simpleKMeans(X,kMax,nIter, plot = False):
    nIterations = nIter
    kMax = kMax
    
    init_centroids = kMeansAdvInitCentroids(X, kMax)
    centroids = init_centroids
    if(plot == True): plotCentroids(X,centroids)
    
    for i in range(1,nIterations):
        idx = findClosestCentroids(X, centroids)
        centroids = findNewCentroids(X,idx,kMax)
        if(plot == True): plotCentroids(X,centroids)
    return [idx,centroids]
    

def kMeansAlg(X,kMax,nIter, nInits, plot = False):    
    
    nInits = nInits
    nIterations = nIter
    kMax = kMax
    centroids = kMeansAdvInitCentroids(X, kMax)
    temp_centroids = centroids
    
    for j in range(0,nInits):    
        for i in range(0,nIterations):
            temp_idx = findClosestCentroids(X, temp_centroids)
            temp_centroids = findNewCentroids(X,temp_idx,kMax)               
        curr_cost = costF(X,temp_idx,temp_centroids)
      
        if (j==0):
            idx = temp_idx
            centroids = temp_centroids
            cost = curr_cost
            
        else:
            if (cost>curr_cost):
                idx = temp_idx
                centroids = temp_centroids
                cost = curr_cost 
        
        if(plot == True): plotCentroids(X,temp_centroids)        
        temp_centroids = kMeansInitCentroids(X, kMax)
    
    if(plot == True): plotCentroids(X,centroids)
    return [idx,centroids]
 
def rebuiltImage(idx,centroid,nrow,ncol):
    image = np.array([])
    image.astype('uint8')
    for i in idx:
        image = np.append(image,np.array(centroid[int(i)].astype('uint8')))
    return image.astype('uint8').reshape(nrow,ncol,len(image)/(ncol*nrow))


def mapImagetoProxy(filename,kMax=16,nIter=5,nInits=1):
    #Option 1
    #data = loadmat('data/ex7/bird_small.mat')
    #A = data['A'] 
    
    #Option 2
    A = cv2.imread(filename)
    plt.imshow(A)
    plt.show()
    dimA_r = len(A)
    dimA_c = len(A[0])
    
    X = np.array([A[i][j] for i in range(0,len(A)) for j in range(0,len(A[1]))])
    kM = kMeansAlg(X,kMax,nIter, nInits)
    proxyA = rebuiltImage(kM[0],kM[1],dimA_r,dimA_c)
    plt.imshow(proxyA)
    plt.show()
    cv2.imwrite('data/ex7/proxyA.png',proxyA)
            
#####
        
data = loadmat('data/ex7/ex7data2.mat')
X = data['X']
simpleKMeans(X,3,5)
kMeansAlg(X,kMax=3,nIter=5,nInits=10)
    
######  

#mapImagetoProxy('data/ex7/bird_small.png',kMax=16,nIter=5,nInits=3)
mapImagetoProxy('data/ex7/boyang.png',kMax=4,nIter=7,nInits=3)


    
