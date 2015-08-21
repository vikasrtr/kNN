"""
kNN - k Nearest Neighbour

__author__: vikas_rtr

"""

import numpy as np
from scipy import stats as sts


class kNN():

    def __init__(self, k):
        self.k = k

    def _euclidian_distance(self, x1, x2):
        """Computes Euclidian Distance b/w two feature vectors
        X1 can be a numpy ndarray and x2 is numpy array
        """
        a= x1-x2
        a2 = a**2
        b = np.sum(a2, axis=1)
        c = np.sqrt(b)
        return c
#         return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def fit(self, X, y):
        """takes input of features and corresponding labels

        """
        self.X_data = X
        self.y = y

    def predict(self, X):
        """Classify features according to euclidian_distance from all data points

        Parameters:

        X:
        numpy ndarray

        """

        Xn = np.copy(X)

        preds = []
        # compute distance from all points
        for x1 in Xn:
            dist = self._euclidian_distance(self.X_data, x1)
            dist = np.vstack((dist, self.y)).T
            dist = dist[dist[:, 0].argsort(axis=0)][:,-1]
            # get a vote from top k
            pred = sts.mode(dist[0:self.k])[0][0]
            preds.append(pred)

        return np.array(preds)
