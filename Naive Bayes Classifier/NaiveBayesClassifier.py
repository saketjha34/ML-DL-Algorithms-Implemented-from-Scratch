import numpy as np

class GaussianNB():
    def __init__(self,type :str = 'binary'):
        self.type= type

    def _softmax(self,x):
        numerator = np.exp(x)
        denominator = np.sum(np.exp(x))
        return numerator/denominator
    
    def _sigmoid(self, x) :
        return 1/(1+np.exp(x))
    
    def score(self,y_true , y_pred):
        return np.float(np.sum(y_true == y_pred)/ len(y_true))
    
    def fit(self,X,y):
        n_samples , n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes , n_features) , dtype=np.float64 )
        self._var = np.zeros((n_classes , n_features) , dtype=np.float64 )
        self._priors = np.zeros(n_classes ,dtype=np.float64 )

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def predict_proba(self, X):
        posteriors = []
        
        for x in X:
            class_posteriors = []
            
            for idx, c in enumerate(self._classes):
                prior = np.log(self._priors[idx])
                posterior = np.log(self._gaussian(idx, x))
                posterior = np.sum(posterior) + prior
                class_posteriors.append(posterior)
            posteriors.append(class_posteriors)

        posteriors = np.array(posteriors)

        if self.type == 'multiclass':
            return self._softmax(posteriors)
        elif self.type == 'binary':
            return self._sigmoid(posteriors)


    def _gaussian(self, class_idx , x):
        mean  = self._mean[class_idx]
        var  = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict(self , x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._gaussian(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)
        
        return self._classes[np.argmax(posteriors)]
    

