from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X , y = make_regression(n_samples=2000,
                        n_features=4, 
                        noise=1,
                        random_state=69)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=69)

from StochasticGradientDescentRegressor import SGDRegressor

model = SGDRegressor()
model.fit(X_train,y_train)
test_preds = model.predict(X_test)
print(f'y_test : {y_test[:5]} test_preds : {test_preds[:5]}')
print(f'RMSE : {model.root_mean_squared_error(test_preds,y_test)}')

