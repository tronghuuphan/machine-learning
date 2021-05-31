from sklearn import datasets, linear_model
import numpy as np

X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 179, 180, 183]]).T
y = np.array([ 50, 51, 54, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68])

print(X)
print(y)


# fit the model by Linear Regression
regr = linear_model.LinearRegression()
regr.fit(X, y)

# Result
print("Scikit-learn's solution: \nw_1 = {} \nw_0={}".format(regr.coef_[0],regr.intercept_))
