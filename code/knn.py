"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np

import utils
from utils import euclidean_dist_squared


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        n,m = X_hat.shape
        y_hat = np.zeros(n)
        # print(X_hat)
        
        distances = euclidean_dist_squared(X_hat, self.X)
        for i, x in enumerate(distances):
            indices_of_min_k_distances = x.argsort()[:self.k]
            # print(indices_of_min_k_distances)
            # print(x)
            y_hat[i] = utils.mode(self.y[indices_of_min_k_distances])
        return y_hat



        


