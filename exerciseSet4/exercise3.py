import csv

import matplotlib.pyplot as plt
import numpy as np


def main_test():
    dim = []
    testError = []
    trainError = []
    (xtrain, ytrain) = getDatapoints('data/TrainingSet1D.csv')
    (xtest, ytest) = getDatapoints('data/TestSet1D.csv')
    bestDegr = 0
    smallestTestError = 500
    bestW = []
    for d in range(20):
        Xtrain = X(xtrain, d)
        Xtest = X(xtest, d)
        w = w_mle(ytrain, Xtrain)
        errTemp = mse(Xtest, ytest, w)
        trainError.insert(d, mse(Xtrain, ytrain, w))
        testError.insert(d, errTemp)
        dim.insert(d, d)
        if (d == 3 or d == 2):
            print(f"For degree {d} polynomial error is {errTemp}")

        # if-block for deciding the smallest error degree and corresponding w
        if (errTemp < smallestTestError):
            smallestTestError = errTemp
            bestDegr = d
            bestW = np.array(w)

    error(X(xtrain, bestDegr), ytrain, X(xtest, bestDegr), ytest, bestDegr)


    plt.subplot(221)
    plt.gca().set_title("Polynomial Degree vs MSE")
    plt.plot(dim, testError, label="Test Error")
    plt.plot(dim, trainError, label="Training Error")
    plt.axis([0, 20, 0, 20])
    plt.xlabel('Degree')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend()

    plt.subplot(222)
    plt.gca().set_title("Least Err. Polyn. and Train.Data")
    plt.plot(xtrain, ytrain, 'r+')
    plt.axis([-6, 6, -10, 20])
    plot_function(bestW)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)

    plt.subplot(223)
    plt.gca().set_title(f"Polynomial with Degree 3 and Training Data ")
    plt.plot(xtrain, ytrain, 'r+', label="Training Data")
    plt.axis([-6, 6, -10, 20])
    Xtrain = X(xtrain, 3)
    w3 = w_mle(ytrain, Xtrain)
    plot_function(w3, "degr=3")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()

    plt.subplot(224)
    plt.gca().set_title("Deg 1,2,6")
    plt.plot(xtrain, ytrain, 'r+', label="Training Data")
    plt.axis([-6, 6, -10, 20])
    Xtrain = X(xtrain, 6)
    w6 = w_mle(ytrain, Xtrain)
    plot_function(w6, "degr=6")
    Xtrain = X(xtrain, 2)
    w2 = w_mle(ytrain, Xtrain)
    plot_function(w2, "degr=2")
    Xtrain = X(xtrain, 1)
    w1 = w_mle(ytrain, Xtrain)
    plot_function(w1, "degr=1")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()


    plt.show()
    print(f"Best polynomial degree:{bestDegr} with test error:{smallestTestError}")


# loads datapoints from a file into an tuple (x,y) where x and y are arrays/vectors
def getDatapoints(filestring, type=0):
    points = []
    x = []
    y = []
    with open(filestring) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if (type != 0):
                points.insert(line_count, [float(row[0]), float(row[1])])
            else:
                x.insert(line_count, float(row[0]))
                y.insert(line_count, float(row[1]))
            line_count += 1

        print(f'Processed {line_count} lines of {filestring}.')
        if (type != 0):
            return np.array(points)
    return (np.array(x), np.array(y))


# returns the matrix X (exercise 3.a))
def X(vector, d):
    vlen = vector.size
    X = []

    for i in range(vlen):
        vali = vector[i]
        Xi = []
        for j in range(d + 1):
            Xi.append(vali ** j)
        X.insert(i, np.array(Xi))
    return np.array(X)


# exercise 3.b)
def w_mle(y, X):
    A = np.dot(X.transpose(),y)
    return np.dot( np.linalg.inv(np.dot(X.transpose(), X)), A )


#def mse(X, y): #using w_mle
#    return 1/X.shape[1] * np.dot( ( np.dot(X, w_mle(y, X)) - y ).transpose(), (np.dot(X, w_mle(y, X)) - y) )
def mse(X, y, w): #using any w
    return 1/X.shape[1] * np.dot( ( np.dot(X, w) - y).transpose(), (np.dot(X, w) - y) )


#exercie 3.f)
def w_ridge(X, y, lam):
    Y = np.dot(X.transpose(), y)
    W = (lam * np.eye(X.shape[1])) + np.dot(X.transpose(), X)
    Z = np.dot( np.linalg.inv(W), Y)
    return Z


def plot_function(v, g_label=None):
    w = np.flip(v)
    f = np.poly1d(w)
    x = np.linspace(-5, 5, 100)
    if (g_label != None):
        plt.plot(x, f(x), label=g_label)
    else:
        plt.plot(x, f(x))


# method for finding a good combination of d and lambda
def error(Xtrain, ytrain, Xtest, ytest, d):
    err = 100

    for lam in range(1, 11):
        w = w_ridge(Xtrain, ytrain, 1 / lam)
        current_err = mse(Xtest, ytest, w)
        if (current_err < err):
            err = current_err
            current_lam = lam
    print(f"Error in ridge regression: {err} with d = {d} and lambda = {1 / current_lam}")
