# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt #sets up plotting under plt
import random
from scipy.io import loadmat
from sklearn import svm
import math
import scipy.optimize as op
import string
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

f = open('data/ex6/emailSample1.txt', "r")
emailContent = f.read()


def processEmail(emailContent):
    emailContent = emailContent.lower()
    emailContent = re.sub('<[^<>]+>', ' ',emailContent)
    emailContent = re.sub('(http|https)://[^\s]*', 'httpaddr',emailContent)
    emailContent = re.sub('[^\s]+@[^\s]+', 'emailaddr',emailContent)
    emailContent = re.sub('[$]+', 'dollar',emailContent)
    emailContent = re.sub('[0-9]+', 'number', emailContent)
    emailContent = re.sub('[0-9]+', 'number', emailContent)
    emailContent = re.sub('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]',' ',emailContent)
    #thanks! https://github.com/kaleko/CourseraML/blob/master/ex6/ex6_spam.ipynb
    i=0
    emailContent = word_tokenize(emailContent)
    for w in emailContent:
        emailContent[i] = str(ps.stem(w))
        i=i+1       
    return emailContent

def map2Vocab(emailContent):
    emailContent = processEmail(emailContent)
    vocabDict = getVocabDict()
    index = 0
    word_indices = []
    for w in emailContent:
        index = findInDict(vocabDict, w)
        if index >=0:
            word_indices.append(index)     
    return word_indices

def map2FeatureVector(emailContent):
    emailContent = processEmail(emailContent)
    vocabDict = getVocabDict()
    arr = np.zeros((len(vocabDict.keys())))
    for word in emailContent:
        pos = findInDict(vocabDict, word)
        if pos >=0: arr[pos-1] = 1
    return arr

def map2FeatureVector2(emailContent):
    word_indices = map2Vocab(emailContent)
    vocabDict = getVocabDict()
    arr = np.zeros((len(vocabDict.keys())))
    for pos in word_indices:
        arr[pos-1] = 1
    return arr   
    
def getVocabDict():
    with open('data/ex6/vocab.txt') as reader:
        vocab = {}
        for line in reader:
            a=line.split()
            vocab[a[1]]=a[0]
    return vocab
    
def findInDict(dic, key):
    if key in dic:
        return int(dic[key])
    else: return -1
    
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
    
dataTrain = loadmat('data/ex6/spamTrain.mat')
xtrain = dataTrain['X']
ytrain = dataTrain['y']

dataTest = loadmat('data/ex6/spamTest.mat')
xtest = dataTest['Xtest']
ytest = dataTest['ytest']


#Linear
kern='linear'
ytrain = ytrain.flatten()
C=1
#Note that sklearn svm adds the intercept coefficient by itself
svm_= svm.SVC(C,kernel=kern)
svm_.fit(xtrain,ytrain)
accuracy_train = float(sum(svm_.predict(xtrain)==ytrain.ravel()))/float(len(xtrain))
accuracy_test = float(sum(svm_.predict(xtest)==ytest.ravel()))/float(len(xtest))
print("Linear:")
print(accuracy_train)
print(accuracy_test)

#Gaussian

kern='rbf'
ytrain = ytrain.flatten()
C=1
sigma=0.5
#Note that sklearn svm adds the intercept coefficient by itself
svm_= svm.SVC(C,kernel=kern,gamma=math.pow(sigma,-2))
svm_.fit(xtrain,ytrain)
accuracy_train = float(sum(svm_.predict(xtrain)==ytrain.ravel()))/float(len(xtrain))
accuracy_test = float(sum(svm_.predict(xtest)==ytest.ravel()))/float(len(xtest))
print("Gaussian:")
print(accuracy_train)
print(accuracy_test)