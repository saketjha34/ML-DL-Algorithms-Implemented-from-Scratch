import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from StochasticGradientDescentClassifier import SGDClassifier

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

model = SGDClassifier(epochs=1000)
model.fit(X,y)
pred = model.predict(X)
print(pred[:10])
print(y[:10])
pred_proba = model.predict_proba(X)
print(pred_proba[:10])