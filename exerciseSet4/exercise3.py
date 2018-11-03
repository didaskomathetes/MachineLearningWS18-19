import csv

import matplotlib.pyplot as plt
import numpy as np


def main_test():
    dim = []
    error = []
    (xtrain, ytrain) = getDatapoints('data/TrainingSet1D.csv')
    (xtest, ytest) = getDatapoints('data/TestSet1D.csv')
    bestDegr = 0
    smallestError = 500
    bestW = []
    for d in range(20):
        Xtrain = X(xtrain, d)
        Xtest = X(xtest, d)
        w = w_mle(ytrain, Xtrain)
        errTemp = mse(Xtest, ytest, w)
        error.insert(d, errTemp)
        dim.insert(d, d)
        if (errTemp < smallestError):
            smallestError = errTemp
            bestDegr = d
            bestW = np.array(w)

    plt.subplot(221)
    plt.plot(dim, error)
    plt.xlabel('Degree')
    plt.ylabel('Error')
    plt.grid(True)

    plt.subplot(222)
    plt.plot(xtrain, ytrain, 'r+')
    plt.axis([-6, 6, -20, 20])
    plot_function(bestW)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)


    plt.show()
    print(f"Best degree:{bestDegr} with error:{smallestError}")




def getDatapoints(filestring, type=0):
    if (type != 0):
        points = []
    else:
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


def w_mle(y, X):
    return np.dot( np.linalg.inv(np.dot(X.transpose(), X)), np.dot(X.transpose(),y) )


#def mse(X, y): #using w_mle
#    return 1/X.shape[1] * np.dot( ( np.dot(X, w_mle(y, X)) - y ).transpose(), (np.dot(X, w_mle(y, X)) - y) )
def mse(X, y, w): #using any w
    return 1/X.shape[1] * np.dot( ( np.dot(X, w) - y).transpose(), (np.dot(X, w) - y) )


def w_ridge(X, y, lam):
    Y = np.dot(X.transpose(), y)
    W = np.diag(lam) + np.dot(X.transpose(), X)
    Z = np.dot( np.linalg.inv(W), Y)
    return Z


def plot_function(w):
    f = np.poly1d(w)
    x = np.linspace(-5, 5, 100)
    plt.plot(x, f(x))
