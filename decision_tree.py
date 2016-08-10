"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the decision tree classifier with real-values features.
Training algorithm: ID3

Author: Noga Zaslavsky
Date: April, 2016

"""
import numpy as np

def entropy(p):
    if p == 0 or p ==1:
        return 0
    else:
        return -p*np.log2(p)-(1-p)*np.log2(1-p)


class Node(object):
    """ A node in a real-valued decision tree.
        Set all the attributes properly for viewing the tree after training.
    """
    def __init__(self,leaf = True,left = None,right = None,samples = 0,feature = None,theta = 0.5,gain = 0,label = None):
        """
        Parameters
        ----------
        leaf : True if the node is a leaf, False otherwise
        left : left child
        right : right child
        samples : number of training samples that got to this node
        feature : a coordinate j in [d], where d is the dimension of x (only for internal nodes)
        theta : threshold over self.feature (only for internal nodes)
        gain : the gain of splitting the data according to 'x[self.feature] < self.theta ?'
        label : the label of the node, if it is a leaf
        """
        self.leaf = leaf
        self.left = left
        self.right = right
        self.samples = samples
        self.feature = feature
        self.theta = theta
        self.gain = gain
        self.label = label


class DecisionTree(object):
    """ A decision tree for bianry classification.
        Training method: ID3
    """

    def __init__(self,max_depth):
        self.root = None
        self.max_depth = max_depth

    def create_A(self, X, y):
        if len(X) == 0:
            return []
        m, d = X.shape
        A = np.zeros([m+1, d])
        x_sorted = X

        for i in range(d):
            x_sorted = x_sorted[np.argsort(x_sorted[:, i])]

            for j in range(m-1):
                A[j+1][i] = (x_sorted[j][i] + x_sorted[j+1][i])/2.0

        A[0].fill(-float("inf"))
        A[m].fill(float("inf"))
        return A

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """

        A = self.create_A(X, y)
        self.root = self.ID3(X,y,A, 0)


    def ID3(self,X, y, A, depth):
        """
        Gorw a decision tree with the ID3 recursive method

        Parameters
        ----------
        X, y : sample
        A : array of d*m real features, A[j,:] row corresponds to thresholds over x_j
        depth : current depth of the tree

        Returns
        -------
        node : an instance of the class Node (can be either a root of a subtree or a leaf)
        """

        node = Node(samples=len(X))

        if np.all(y == -1):
            node.label = -1
            return node
        if np.all(y == 1):
            node.label = 1
            return node

        if A.size == 0 or depth == self.max_depth:
            majority = 0.0

            for i in range(len(y)):
                majority += y[i]
            if majority >= 0:
                node.label = 1
            else:
                node.label = -1
            return node

        gain = self.info_gain(X, y, A)
        m, d = gain.shape
        max_i, max_j = 0,0
        for i in range (m):
            for j in range (d):
                if gain[i][j] > gain[max_i][max_j]:
                    max_i, max_j = i, j

        X_left_to_theta = np.array([])
        X_right_to_theta = np.array([])

        y_left_to_theta = np.array([])
        y_right_to_theta = np.array([])

        theta = A[max_i][max_j]

        for i in range(len(X)):
            if X[i][max_j] >= theta:
                np.concatenate((X_right_to_theta, X[i]))
                np.concatenate((y_right_to_theta, y[i]))
            else:
                np.concatenate((X_left_to_theta, X[i]))
                np.concatenate((y_left_to_theta, y[i]))


        A_right_theta = self.create_A(X_right_to_theta, y_right_to_theta)
        A_left_theta = self.create_A(X_left_to_theta, y_left_to_theta)

        node = Node(leaf=False, feature= max_j, theta=theta, samples=len(X), gain=gain[max_i][max_j])
        node.left = self.ID3(X_left_to_theta, y_left_to_theta, A_left_theta, depth+1)
        node.right = self.ID3(X_right_to_theta, y_right_to_theta, A_right_theta, depth+1)

        return node

    @staticmethod
    def info_gain(X, y, A):
        """
        Parameters
        ----------
        X, y : sample
        A : array of m*d real features, A[:,j] corresponds to thresholds over x_j

        Returns
        -------
        gain : m*d array containing the gain for each feature
        """
        m, d = A.shape
        gain = np.zeros([m, d])

        p=0.0
        for i in range(len(y)):
            if y[i] == 1:
                p += 1

        p /= len(y)

        entropy_S = entropy(p)

        for i in range(m):
            for j in range(d):
                y_left_theta, y_right_theta= np.array([]), np.array([])
                p_left, p_right = 0.0,0.0
                for a in range(len(X)):
                    for b in range(d):
                        if X[a][b] >= A[i][j]:
                            y_right_theta = np.append(y_right_theta, y[a])
                            if y[a] == 1:
                                p_right += 1
                        else:
                            y_left_theta = np.append(y_left_theta, y[a])
                            if y[a] == 1 :
                                p_left += 1
                if len(y_right_theta) == 0:
                    p_right = 0.0
                else:
                    p_right /= len(y_right_theta)
                if len(y_left_theta) == 0:
                    p_left = 0.0
                else:
                    p_left /= len(y_left_theta)

                gain[i][j] = entropy_S -p_right*entropy(p_right) - p_left * entropy(p_left)

        return gain


    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """

        y_hat = np.zeros(len(X))

        for i in range(len(X)):
            curr = self.root
            while curr.leaf == False:
                if X[i][curr.feature] >= curr.theta:
                    curr = curr.right
                else:
                    curr = curr.left
            y_hat[i] = curr.label

        return y_hat



    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        y_hat = self.predict(X)
        sum_err = 0.0
        for i in range(len(y)):
            if y[i] != y_hat[i]:
                sum_err += 1


        return sum_err/len(y)