
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)
from NaiveBayesClassifier import GaussianNB

clf = GaussianNB(type = 'multiclass')
clf.fit(X_train, y_train)
predictions = clf.predict(X_train)
print("Naive Bayes Classification Accuracy :", accuracy_score(y_train, predictions))
print(predictions[:10])
print(y_train[:10])
print(X_train.shape)
print(len(clf.predict_proba(X_train)))
print(clf.predict_proba(X_train))