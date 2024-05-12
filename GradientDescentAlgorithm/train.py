import numpy as np

from GradientDescentAlgorithim import GradientDescent
# Known Weights and bias
w = np.array([2,4])
b = 2

# Dataset
X = np.random.randn(100, 2)
y = np.dot(X,w) + b


tol = 1e-3
epochs = 900
learning_rate = 0.001

GradientDescent(X,y,epochs,learning_rate,tol)