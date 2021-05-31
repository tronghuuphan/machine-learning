import numpy as np
from sklearn import neighbors, datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Define my weight
def myweight(distances):
    sigma2 = 0.4
    return np.exp(-distances**2 / sigma2)

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print('Labes:', np.unique(iris_y))

# Split train and test data
np.random.seed(7)
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=130)
print('Train size:', X_train.shape, 'Test size: ', X_test.shape)

# 1NN
model = neighbors.KNeighborsClassifier(n_neighbors=1, p=2, weights=myweight) # p=2 -> L2 norm
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy of 1NN:' ,(100*accuracy_score(y_test, y_pred)))

# 7NN
model =  neighbors.KNeighborsClassifier(n_neighbors=7, p=2, weights=myweight)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy of 7NN: ' ,(100*accuracy_score(y_test, y_pred)))
