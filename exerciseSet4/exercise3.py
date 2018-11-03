import csv

import numpy as np


def main_test(d):
    # print("Output Matrix:")
    # print(X(np.array([3, 2, 4, 8]), 3))

    (x, y) = getDatapoints('data/TrainingSet1D.csv')
    Xm = X(x, d)
    w = w_mle(y, Xm)

    print(w)


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
    return 1/X.shape[1] * np.dot( ( np.dot(X, w) - y) ).transpose(), (np.dot(X, w) - y) )


def w_ridge(X, y, lam):
    Y = np.dot(X.transpose(), y)
    W = np.diag(lam) + np.dot(X.transpose(), X)
    Z = np.dot( np.linalg.inv(W), Y)
    return Z
