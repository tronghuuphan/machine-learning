import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('ex1data1.txt',header=None).values
X = data[:,0].reshape(-1,1)
y = data[:,1].reshape(-1,1)


class linearRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.m = y.shape[0]
        self.theta = np.array([[0.],[0.]])
        self.Xc = np.concatenate((np.ones((self.m,1)),X), axis=1)

    def plotData(self):
        plt.plot(self.X, self.y,'rx')
        plt.xlabel('Population')
        plt.ylabel('Profit')
        plt.legend(['Traing Data'])
    def plotLinearFit(self):
        self.plotData()
        plt.plot(self.X, np.dot(self.Xc,self.theta))
        plt.legend(['Training Data','Linear Regression'])
    def computeCost(self):
        return (1/(2*self.m))*np.sum((np.dot(self.Xc,self.theta) - self.y)**2)
    def gradientDescent(self, alpha, iteration):
        for i in range(iteration):
            A = np.sum(np.dot(self.Xc,self.theta) - self.y)
            B = np.sum((np.dot(self.Xc,self.theta) -self.y)*self.X)
            self.theta[0] -= alpha*(1/self.m)*A
            self.theta[1] -= alpha*(1/self.m)*B
            #For Debug
            #print('Cost[{}]: {}'.format(i,self.computeCost()))
        return self.theta




A = linearRegression(X, y)
print('Plot data...')
A.plotData()
plt.show()
print('='*25)
print('Running Gradient Decent...')
print(A.gradientDescent(0.01,1500))
result = A.theta
print('='*25)
print('Plot the linear fit...')
A.plotLinearFit()
plt.show()

