#@ Implementation of perceptron
import numpy as np
class MyPerceptron:
    def __init__(self, learning_rate = 0.1, n_iter = 1000):
        self.lr = learning_rate
        self.epoch = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(self.epoch):
            for i in range(X.shape[0]):
                y_pred = np.dot(self.weights, X[i]) + self.bias                    # calculating the predicted value of y
                self.weights = self.weights + self.lr * (y[i] - y_pred) * X[i]     # updating the weights
                self.bias = self.bias + self.lr * (y[i] - y_pred)                  # updating bias
    print("|--------- TRAINING IS COMPLETED ---------|")
    
    #@ calculating activation function
    def activation_function(self, activation):
        if activation >= 0:
            return 1 
        else:
            return 0
    
    #@ Predicting the data
    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            y_pred.append(self.activation_function(np.dot(self.weights.X[i]) + self.bias))
        return np.array(y_pred)