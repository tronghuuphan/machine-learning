import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as op
data = pd.read_csv('ex2data1.txt',header=None).values
X = data[:,[0,1]]
y = data[:,2].reshape(-1,1)

class logisticRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.m = y.shape[0]
        self.theta = np.zeros((X.shape[1]+1,1))
        self.Xc = np.concatenate((np.ones((self.m,1)), X), axis=1)

    def plotData(self):
        pos = [i for i in range(self.m) if self.y[i] == 1]
        neg = [i for i in range(self.m) if self.y[i] == 0]
        plt.scatter(self.X[pos,0],self.X[pos,1],marker='o',s=10,color='green')
        plt.scatter(self.X[neg,0],self.X[neg,1],marker='x',s=10,color='red')
        plt.legend(['Pass','Fail'])
        plt.xlabel('Exam 1 Score')
        plt.ylabel('Exam 2 score')
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    def costFunction(self,alpha,iteration):
        grad = np.zeros(self.theta.shape)
        h = self.sigmoid(np.dot(self.Xc,self.theta))
        J = (1/self.m)*np.sum(-self.y*np.log(h)-(1-self.y)*np.log(1-h))
        for i in range(iteration):
            h = self.sigmoid(np.dot(self.Xc,self.theta))
            a = np.sum(h-self.y)
            B = np.sum((h-self.y)*self.Xc[:,1].reshape(-1,1))
            C = np.sum((h-self.y)*self.Xc[:,2].reshape(-1,1))
            self.theta[0] -= (alpha/self.m)*a
            self.theta[1] -= (alpha/self.m)*B
            self.theta[2] -= (alpha/self.m)*C
#            print((1/self.m)*np.sum(-self.y*np.log(h)-(1-self.y)*np.log(1-h)))
        print('='*25)
        print(self.theta)




A = logisticRegression(X,y)

A.costFunction(0.02,2000000)
