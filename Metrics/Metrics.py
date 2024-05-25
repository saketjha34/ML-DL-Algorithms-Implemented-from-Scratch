import numpy as np

def root_mean_squared_error(y_true , y_pred):
    return np.sqrt(np.mean(y_true - y_pred)**2)

def mean_squared_error(y_true, y_pred):
    return np.mean(y_true - y_pred)**2

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean(x):
    return np.mean(x)

def mean_deviation(x):
    return np.mean(np.abs(x-np.mean(x)))

def variance(x):
    return np.mean((x - np.mean(x))**2)

def standard_deviation(x):
    return np.sqrt(np.mean((x-np.mean(x))**2))

def r_2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0,1]
    return corr**2

def max_error(y_true , y_pred):
    return np.max(np.abs(y_true - y_pred))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def accuracy_score(y_true , y_pred):
    return np.sum(y_pred==y_true)/len(y_true)

def precision_score(y_true , y_pred):
    return 



