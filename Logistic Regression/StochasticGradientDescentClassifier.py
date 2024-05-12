import numpy as np

class SGDClassifier():

    def __init__(self,verbose =False,learning_rate= 0.01,epochs=1000,batch_size=32) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.verbose = verbose
        self.weights = None
        self.bias = None


    def sigmoid(self,x):
        return 1/(1+ np.exp(-x))
    
    def predict(self, X):
        liner_pred = np.dot(X, self.weights) + self.bias
        sig_pred =  self.sigmoid(liner_pred)
        return [0 if x<=0.5 else 1 for x in sig_pred]
    
    def accuracy(self,y_true , y_pred):
        return np.sum(y_pred==y_true)/len(y_true)
    
    def predict_proba(self,X):
        liner_pred = np.dot(X, self.weights) + self.bias
        sig_pred =  self.sigmoid(liner_pred)
        return sig_pred
    
    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                linear_pred = np.dot(X_batch, self.weights) + self.bias
                predictions = self.sigmoid(linear_pred)

                gradient_weights = (1/n_samples) * np.dot(X_batch.T, (predictions - y_batch))
                gradient_bias = (1/n_samples) * np.sum(predictions-y_batch)

                self.weights -= self.learning_rate * gradient_weights
                self.bias -= self.learning_rate * gradient_bias

            if epoch % 100 == 0:
                y_pred = self.predict(X)
                if not self.verbose:
                    print(f"Epoch: {epoch} | Train Accuracy : {self.accuracy(y,y_pred)}  ")


        return "Model Trained"
    
    def __repr__(self) -> str:
        return f'SGDClassifier(learning_rate = {self.learning_rate} ,tolerance = {self.tolerance} ,batch_size = {self.batch_size} ,epochs = {self.epochs})'

    def __str__(self):
        return self.__repr__()
    
    def get_model_weights_bias(self):
        return self.weights,self.bias