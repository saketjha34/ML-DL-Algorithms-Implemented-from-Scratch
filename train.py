from StochasticGradientDescentRegressor import SGDRegressor
import numpy as np

w = np.array([2,4])
b = 2
model = SGDRegressor()

X = np.random.randn(100, 2)
y = np.dot(X,w) + b
model.fit(X,y)

test_preds = model.predict(np.array([1,3]))
print(test_preds)
print(model.get_model_weights_bias())
print(model.__str__())