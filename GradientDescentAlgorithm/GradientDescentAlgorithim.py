import numpy as np

def GradientDescent(X,y,epochs,learning_rate,tol):
    
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    print(f"Epoch: -1 | MAE Train Loss: None current Weight -> {weights} ,current bias -> {bias} ")
    for epoch in range(epochs):

        y_predicted = np.dot(weights,X.T) + bias
        
        error = y_predicted[0] - y
        loss = np.mean((y - y_predicted)**2)

        gradient_weights = np.dot(X.T, error) / X.shape[0]
        gradient_bias = np.mean(error)

        weights = weights - learning_rate * gradient_weights
        bias = bias - learning_rate * gradient_bias

        print(f"Epoch: {epoch} | MAE Train Loss: {loss} current Weight -> {weights} ,current bias -> {bias} ")

        if np.linalg.norm(gradient_weights) < tol:
            print("Convergence reached.")
            break

