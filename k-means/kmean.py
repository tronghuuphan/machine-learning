import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist # Thu vien ho tro tinh khoang cach

np.random.seed(18)

means = [[2, 2], [8, 3], [3, 6]]
cov = [[2, 0], [0, 2]]
n_sample = 500
n_cluster = 3

X0 = np.random.multivariate_normal(means[0], cov, n_sample)
X1 = np.random.multivariate_normal(means[1], cov, n_sample)
X2 = np.random.multivariate_normal(means[2], cov, n_sample)
X = np.concatenate((X0, X1, X2), axis=0)


def kmeans_display(X, pred_label):
    plt.xlabel('x')
    plt.ylabel('y')
    for i in range(n_sample*n_cluster):
        if pred_label[i]==0:
            plt.plot(X[i,0], X[i,1], 'bo', markersize=2)
        elif pred_label[i]==1:
            plt.plot(X[i,0], X[i,1], 'ro', markersize=2)
        elif pred_label[i]==2:
            plt.plot(X[i,0], X[i,1], 'go', markersize=2)
    plt.plot()
    plt.show()

# Use scikit-learn lib

model = KMeans(n_clusters=n_cluster, random_state=0)
model.fit(X)

print('Centers found by KMeans: ')
print(model.cluster_centers_)

pred_label = model.predict(X)
kmeans_display(X, pred_label)
