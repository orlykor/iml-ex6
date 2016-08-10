"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the k nearest neighbors classifier.

Author: Noga Zaslavsky
Date: April, 2016

"""
import numpy as np

class kNN(object):

    def __init__(self, k):
        self.k = k
        self.y = None
        self.X = None
        return


    def find_max_dis(self, distances):
        is_max = 0
        for i in range(len(distances)):
            if (distances[i][0] > distances[is_max][0]):
                is_max = i
        return is_max


    def train(self, X, y):
        self.X = X
        self.y = y

        """
        Train this classifier over the sample (X,y)
        """


    def predict(self, X):

        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        y_hat = np.zeros(len(X))
        for j in range(len(X)):
            distances = []
            for i in range(len(self.X)):
                dis = np.linalg.norm(self.X[i]-X[j])**2
                if len(distances) < self.k:
                    distances.append((dis,self.y[i]))
                else:
                    max = self.find_max_dis(distances)
                    if distances[max][0] > dis:
                        distances[max] = (dis, self.y[i])
            sum = 0
            for tup in distances:
                sum += tup[1]

            if sum >= 0:
                y_hat[j] = 1
            else:
                y_hat[j] = -1

        return y_hat

    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        y_hat = self.predict(X)
        num_err = 0.0
        for i in range(len(X)):
            if y[i] != y_hat[i]:
                num_err += 1

        return num_err/len(X)


