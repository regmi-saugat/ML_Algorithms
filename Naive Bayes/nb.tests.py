import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets         # creating and loading the datasets
import matplotlib.pyplot as plt

from nb import NaiveBayes


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)  # spliting the data into training and testing samples

nb = NaiveBayes()           # creating the naivebayes classifier, which is imported from the file 'nb.py'
nb.fit(X_train, y_train)    # fitting the training data and training lables
predictions = nb.predict(X_test)    # getting prediction for test samples

print("Naive Bayes classificaiton accuracy", accuracy(y_test, predictions))    # calculating the accuracy

