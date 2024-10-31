# ML-DL-Algorithms-Implemented-from-Scratch

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Mathematical Foundations and Articles](#mathematical-foundations-and-articles)
- [Installation](#installation)
- [Usage](#usage)
- [Implementation](#implementation)
- [Model Architecture](#model-architecture)
- [Deployment](#deployment)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

This repository contains implementations of several fundamental machine learning algorithms from scratch. Each implementation focuses on understanding the core mathematical principles and the step-by-step processes involved in these algorithms. The algorithms implemented include Support Vector Machine (SVM), Logistic Regression, Linear Regression, Naive Bayes, K-Nearest Neighbors (KNN), Gradient Descent.

## Project Structure

```bash
ML-DL-Algorithms-Implemented-from-Scratch/
├── GradientDescentAlgorithm/
├── KNearestNeighbour/
├── LinearRegression/
├── LogisticRegression/
├── Metrics/
├── NaiveBayesClassifier/
├── SupportVectorMachine/
├── .gitattributes
├── .gitignore
├── requirements.txt
├── LICENSE
├── README.md
```

### Mathematical Foundations and Articles

1. **Support Vector Machine (SVM)**
   - **Introduction**: SVMs are supervised learning models used for classification and regression. They work by finding the hyperplane that best divides a dataset into classes.
   - **Mathematics**: The optimization problem involves maximizing the margin between different classes, which can be solved using Lagrange multipliers.
   - **References**:
     - [Understanding SVM](https://scikit-learn.org/stable/modules/svm.html#svm)
     - [SVM Mathematical Foundations](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)

2. **Logistic Regression**
   - **Introduction**: Logistic Regression is used for binary classification problems. It models the probability that a given input belongs to a particular class.
   - **Mathematics**: It uses the logistic function to model a binary outcome, and parameters are typically estimated using Maximum Likelihood Estimation (MLE).
   - **References**:
     - [Logistic Regression Explained](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc)
     - [Mathematics of Logistic Regression](https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch12.pdf)

3. **Linear Regression**
   - **Introduction**: Linear Regression is a linear approach to modeling the relationship between a dependent variable and one or more independent variables.
   - **Mathematics**: It aims to minimize the sum of squared differences between the observed and predicted values.
   - **References**:
     - [Introduction to Linear Regression](https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/01/lecture-01.pdf)
     - [Linear Regression Mathematics](https://towardsdatascience.com/a-step-by-step-explanation-of-principal-component-analysis-pca-45e86c85e2e3)

4. **Naive Bayes**
   - **Introduction**: Naive Bayes classifiers are a family of simple probabilistic classifiers based on Bayes' theorem with strong independence assumptions between features.
   - **Mathematics**: It calculates the probability of a class given a set of features using the product of individual probabilities.
   - **References**:
     - [Naive Bayes Classifier Overview](https://scikit-learn.org/stable/modules/naive_bayes.html)
     - [Mathematical Background](https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/)

5. **K-Nearest Neighbors (KNN)**
   - **Introduction**: KNN is a non-parametric method used for classification and regression. It classifies a sample based on the majority class among its k nearest neighbors.
   - **Mathematics**: Distance metrics (like Euclidean distance) are used to find the nearest neighbors.
   - **References**:
     - [KNN Algorithm Explained](https://towardsdatascience.com/k-nearest-neighbors-knn-algorithm-for-machine-learning-e883219c8f26)
     - [Mathematics of KNN](https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/)

6. **Gradient Descent**
   - **Introduction**: Gradient Descent is an optimization algorithm used to minimize the cost function in machine learning models.
   - **Mathematics**: It iteratively updates model parameters in the direction of the negative gradient of the cost function.
   - **References**:
     - [Gradient Descent Explained](https://towardsdatascience.com/gradient-descent-algorithm-and-its-variants-10f652806a3)
     - [Mathematical Details](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html)


## Installation
To set up the project, clone the repository and ensure you have the required Python packages installed:

```bash
git clone https://github.com/saketjha34/ML-DL-Algorithms-Implemented-from-Scratch.git
cd ML-DL-Algorithms-Implemented-from-Scratch
pip install -r requirements.txt  
```

## Usage
Navigate to the respective algorithm directory and run the implementation scripts. For example, to test the Logistic Regression implementation, navigate to the `Logistic Regression` directory and run the script:

```bash
cd Logistic Regression
python train.py
```

## Implementation 

```python

from Logistic_Regression.StochasticGradientDescentClassifier import SGDClassifier
clf = SGDClassifier(verbose =False, learning_rate= 0.01, epochs=1000, batch_size=32)

```

```python

from Naive_Bayes_Classifier.NaiveBayesClassifier import GaussianNB
clf = GaussianNB()

```

```python
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from Naive_Bayes_Classifier.NaiveBayesClassifier import GaussianNB
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
```

```bash
cd Support Vector Machine
python train.py
python svm.py
```

## Model Architecture
Each algorithm is implemented from scratch, focusing on the core mathematical operations and steps required. This includes data preprocessing, model training, and evaluation procedures.

## Deployment
This project focuses on the foundational understanding of ML and DL algorithms and does not include deployment scripts. Users are encouraged to integrate these algorithms into their own applications and pipelines as needed.

### References
- **Support Vector Machine (SVM)**
  - [Understanding SVM](https://scikit-learn.org/stable/modules/svm.html#svm)
  - [SVM Mathematical Foundations](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)

- **Logistic Regression**
  - [Logistic Regression Explained](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc)
  - [Mathematics of Logistic Regression](https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch12.pdf)

- **Linear Regression**
  - [Introduction to Linear Regression](https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/01/lecture-01.pdf)
  - [Linear Regression Mathematics](https://towardsdatascience.com/a-step-by-step-explanation-of-principal-component-analysis-pca-45e86c85e2e3)

- **Naive Bayes**
  - [Naive Bayes Classifier Overview](https://scikit-learn.org/stable/modules/naive_bayes.html)
  - [Mathematical Background](https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/)

- **K-Nearest Neighbors (KNN)**
  - [KNN Algorithm Explained](https://towardsdatascience.com/k-nearest-neighbors-knn-algorithm-for-machine-learning-e883219c8f26)
  - [Mathematics of KNN](https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/)

- **Gradient Descent**
  - [Gradient Descent Explained](https://towardsdatascience.com/gradient-descent-algorithm-and-its-variants-10f652806a3)
  - [Mathematical Details](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html)

## Contributing

Contributions are welcome! If you have any improvements or new models to add, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/saketjha34/ML-DL-Algorithms-Implemented-from-Scratch/blob/main/LICENSE) file for details.

## Contact
For any questions or feedback, please contact [Saket Jha](saketjha0324@gmail.com).
