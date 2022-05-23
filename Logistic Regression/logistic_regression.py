import numpy as np
from sklearn import linear_model


class LogisticRegression:

    '''
    Learning rate; basically it is very small
    n_iters = it is the number of iteration; it is deafult of 1000; it will determine how many iteration we use for the gradient descent
    '''

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    # defining the fit method; by following the convention of scikit-learn library
    '''
    X = training samples
    y = training labels
    It will involve the training step and gradient descent step'''

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Init the parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            # applying the linear model
            linear_model = np.dot(X, self.weights) + self.bias
            # defining the predict method
            y_predicted = self._sigmoid(linear_model)

            # computing the gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # now, for updating the parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    # Here, we get the test samples that we want to predict
    def predict(self, X):
        # applying the linear model
        linear_model = np.dot(X, self.weights) + self.bias
        # defining the predict method
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
