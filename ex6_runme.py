"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Running script for Ex6.

Author:
Date: April, 2016

"""
from adaboost import AdaBoost
from nearest_neighbors import kNN
from ex6_tools import*
from decision_tree import *
# import numpy as np




x_train = np.loadtxt('SynData/X_train.txt')
y_train = np.loadtxt('SynData/y_train.txt')

x_val = np.loadtxt('SynData/X_val.txt')
y_val = np.loadtxt('SynData/y_val.txt')

x_test = np.loadtxt('SynData/X_test.txt')
y_test = np.loadtxt('SynData/y_test.txt')

# plot_graph(x, y, val_err, train_err, name):



def Q3(): # AdaBoost
    T = [1,5,10,50,100,200]
    T_loop = [1,5,10]
    train_err = []
    valid_err = []

    plt.figure("decisions of the learned classifiers for T")
    num_graph = 0
    for i in range(3,41):
        T_loop.append(i*5)

    for t in T_loop:
        ada_boost = AdaBoost(DecisionStump, t)
        ada_boost.train(x_train, y_train)
        if (t in T):
            num_graph += 1
            plt.subplot(3,2, num_graph)
            decision_boundaries(ada_boost, x_train, y_train, "T = %d" %t)

        train_err.append(ada_boost.error(x_train, y_train))
        valid_err.append(ada_boost.error(x_val, y_val))

    plt.figure("training error and the validation error")
    plt.plot(T_loop, train_err, 'ro-', hold=False, label= "Training Error")
    plt.plot(T_loop, valid_err, 'go-', label= "Validation Error")
    plt.legend()
    plt.show()

    '''
    find the T min, and plot it with training error
    '''

    plt.figure("decision boundaries of T min, with the training data")

    T_hat = 5 * np.argmin(valid_err)
    ada_boost = AdaBoost(DecisionStump, T_hat)
    ada_boost.train(x_train, y_train)
    test_err = ada_boost.error(x_test, y_test)
    decision_boundaries(ada_boost, x_train, y_train, "T = %d" %T_hat)
    plt.show()
    print ("The value of T that minimizes the validation error is: ", T_hat)
    print("the test error of the corresponding classifier is: ", test_err)


    return

def Q4(): # decision trees

    max_depth = [1,2,3,4,5,6,7,8,9,10,11,12]
    num_graph = 0
    train_err = []
    valid_err = []
    plt.figure("Decision tree: decisions of the learned classifiers for max_depth")

    for d in max_depth:
        num_graph += 1
        d_tree = DecisionTree(d)
        d_tree.train(x_train, y_train)
        plt.subplot(3,4, num_graph)
        decision_boundaries(d_tree, x_train, y_train, "Max depth= %d" %d)

        train_err.append(d_tree.error(x_train, y_train))
        valid_err.append(d_tree.error(x_val, y_val))

    plt.figure("Decision tree: training error and validation error as a function of max_depth")
    plt.plot(max_depth, train_err, 'ro-', hold=False, label= "Training Error")
    plt.plot(max_depth, valid_err, 'go-', label= "Validation Error")
    plt.legend()

    max_depth_val = np.argmin(valid_err)
    d_hat = max_depth[max_depth_val]
    d_tree = DecisionTree(d_hat)
    d_tree.train(x_train, y_train)
    test_err = d_tree.error(x_test, y_test)

    print("The value of max depth that minimizes the validation error is: ", d_hat)
    print("The test error of the corresponding classifier is: ", test_err)

    # The value of T that minimizes the validation error is:  55
    # the test error of the corresponding classifier is:  0.184

    return

def Q5(): # kNN
    K = [1, 3, 10 ,100, 200, 500]
    num_graph = 0
    plt.figure("kNN: decisions for each k")
    train_err = []
    valid_err = []

    for k in K:
        num_graph += 1
        knn = kNN(k)
        knn.train(x_train, y_train)
        plt.subplot(3,2, num_graph)
        decision_boundaries(knn, x_train, y_train, "K = %d" %k)

        train_err.append(knn.error(x_train, y_train))
        valid_err.append(knn.error(x_val, y_val))

    plt.figure("kNN: training error and the validation error as a function of log(k)")
    plt.plot(np.log(K), train_err, 'ro-', hold=False, label= "Training Error")
    plt.plot(np.log(K), valid_err, 'go-', label= "Validation Error")
    plt.legend()

    index_k_hat = np.argmin(valid_err)
    k_hat = K[index_k_hat]
    knn = kNN(k_hat)
    knn.train(x_train, y_train)
    test_err = knn.error(x_test, y_test)

    print("The value of K that minimizes the validation error is: ", k_hat)
    print("The test error of the corresponding classifier is: ", test_err)

    '''
    The value of K that minimizes the validation error is:  10
    the test error of the corresponding classifier is:  0.084
    '''
    plt.show()

    return

def Q6(): # Republican or Democrat?

    feature_names = np.loadtxt("CongressData/feature_names.txt", dtype=bytes).astype(str)
    class_names = np.loadtxt("CongressData/class_names.txt", dtype=bytes).astype(str)
    X = np.loadtxt("CongressData/votes.txt")
    y = np.loadtxt("CongressData/parties.txt")

    #Split randomly the data into training (50%), validation (40%) and test (10%) sets
    X_train_congress = X[:0.5 * len(X), :]
    X_val_congress = X[0.5 * len(X):0.9 * len(X), :]
    X_test_congress = X[0.9 * len(X):, :]

    y_train_congress = y[:0.5 * len(X)]
    y_val_congress = y[0.5 * len(X):0.9 * len(X)]
    y_test_congress = y[0.9 * len(X):]

    # # I ran the decision tree classifier with a lot of options of max depth,
    # # and as it performed in plot "errors congress decisionTree" it is the
    # # first option that minimize the validation error
    max_depth = 5

    # I ran the AdaBoost classifier with a lot of options of T, and as it
    # performed in plot "errors congress adaBoost" it is the first option that
    # minimize the validation error
    T = 33

    # I ran the k-nearest neighbors classifier with a lot of options of k,
    # and as it performed in plot "errors congress kNN" it is the first option
    # that minimize the validation error
    k = 1

    decisionTree = DecisionTree(max_depth)
    adaBoost = AdaBoost(DecisionStump, T)
    knn = kNN(k)

    classifiers = (decisionTree, adaBoost, knn)
    validation_errors, test_errors  = {}, {}

    for classifier in classifiers:
        classifier.train(X_train_congress, y_train_congress)
        validation_errors[type(classifier)] = classifier.error(X_val_congress, y_val_congress)
        test_errors[type(classifier)] = classifier.error(X_test_congress, y_test_congress)

    decisionTree, adaBoost, knn = classifiers
    print("The validation error for the decision tree with %d as max_depth is"
          " %1.3f, and the test error is %1.3f" % (
              max_depth, validation_errors[type(decisionTree)],
              test_errors[type(decisionTree)]))

    print("The validation error for the k-nearest neighbors with %d as k is"
          " %1.3f, and the test error is %1.3f" % (
              k, validation_errors[type(knn)],
              test_errors[type(knn)]))

    print("The validation error for the adaBoost with %d as T is"
          " %1.3f, and the test error is %1.3f" % (
              T, validation_errors[type(adaBoost)],
              test_errors[type(adaBoost)]))

    return

if __name__ == '__main__':
    # TODO - run your code for questions 3-6
    Q3()
    # Q5()
    # Q4()
    pass