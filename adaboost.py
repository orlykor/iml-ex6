"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Noga Zaslavsky
Date: April, 2016

"""
import numpy as np


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights



    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        D = np.empty(len(X))
        D.fill(1/len(X))
        for t in range(self.T):
            self.h[t] = self.WL(D, X, y)
            est = 0
            predict_x = self.h[t].predict(X)
            for i in range(len(X)):
                if(predict_x[i] != y[i]):
                    est += D[i]
            self.w[t] = 0.5*np.log((1/est)-1)
            n_factor = 0
            for j in range(len(X)):
                n_factor += D[j]*np.exp(-self.w[t]*y[j]*predict_x[j])
            for i in range(len(X)):
                D[i] = D[i]*np.exp(-self.w[t]*y[i]*predict_x[i])/n_factor




    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        predictions = [None] *self.T
        for t in range(self.T):
            predictions[t] = self.h[t].predict(X)
        y_hat = np.zeros(len(X))
        for i in range(len(X)):
            sum = 0
            for t in range(self.T):
                sum += self.w[t]*predictions[t][i]

            y_hat[i] = np.sign(sum)
        return y_hat


    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        err = [0]*len(X)
        y_hat = self.predict(X)

        for i in range(len(X)):
            if (y_hat[i] != y[i]):
                err[i] = 1

        return np.mean(err)