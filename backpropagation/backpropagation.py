import numpy as np
import pandas as pd

# Define Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define Sigmoid Derivative function
def sigmoid_derivative(x):
    return x*(1-x)

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.layers = layers
        self.alpha = alpha
        self.W = []
        self.b = []

        for i in range(0, len(layers) -1)
            w_ = np.random.randn(layers[i], layers[i+1])
            b_ = np.zeros((layer[i+1], 1))
            self.W.append(w_/layers[i])
            self.b.append(b_)

    def __repr__(self):
        return "Neural network [{}]".format('-'.join(str(l) for l in self.layers))

    def fit_partial(self, x, y):
        A = [x]
        out = A[-1]
        for i in range(0, len(self.layers) - 1):
            out = sigmoid(np.dot(out, self.W[i]) + (self.b[i].T))
            A.append(out)
        y = y.reshape(-1, 1)
        dA = [-(y/A[-1] - (1-y)/(1-A[-1]))]
        dW = []
        db = []
        for i in reversed(range(0, len(self.layers) -1 )):
            dw_ = np.dot((A[i]).T, dA[-1] * sigmoid_derivate(A[i+1]))
            db_ = (np.sum(dA[-1] * sigmoid_derivate(A[i+1])))

