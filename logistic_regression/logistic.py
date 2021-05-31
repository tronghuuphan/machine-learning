import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Load data
data = pd.read_csv('dataset.csv').values
N, d = data.shape
x = data[:, :d-1].reshape(-1,d-1)
y = data[:, 2].reshape(-1,1)

# Visualize data
plt.scatter(x[:10, 0], x[:10, 1], c='red', s=30, label='Cho vay')
plt.scatter(x[10:, 0], x[10:, 1], c='blue', s=30, label='Tu choi')
plt.legend(loc=1)
plt.xlabel('Muc luong')
plt.ylabel('Kinh nghiem')


x = np.hstack((np.ones((N,1)), x))
w = np.array([0., 0.1, 0.1]).reshape(-1,1)

num_of_iteration = 1000
cost = np.zeros((num_of_iteration, 1))
learning_rate = 0.01

for i in range(1, num_of_iteration):
    y_predict = sigmoid(np.dot(x, w))
    cost[i] = -np.sum(np.multiply(y, np.log(y_predict)) + np.multiply(1-y, np.log(1-y_predict)))
    w = w - learning_rate*np.dot(x.T, y_predict-y)
    print(cost[i])

t = 0.5
print(w)


plt.show()
