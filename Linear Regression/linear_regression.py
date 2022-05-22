import numpy as np

# '''defining a class named LinearRegression and storing in the init method'''
class LinearRegression:

    def __init__(self, lr=0.01, n_iters=10000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None


# '''
# # defining the fit method which takes training samples and labels for them which involves training steps and gradient descent
# # for gradient descent: we need some initialization
# '''
    def fit(self, X, y):
        # Init the parameter
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

    # gradient descent is an iterative process so, we use for loop
        for _ in range(self.n_iters):
            # Defining the approximation
            y_predicted = np.dot(X, self.weights) + self.bias

            # Now, we can calculate the derivative of the weights
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            # Similarly calculating the derivative the of bias
            db = (1/n_samples) * np.sum(y_predicted - y)

            # For updating the weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    # '''defining the predict method, when it get's new tests samples it can approximate the values and return the values'''
    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
