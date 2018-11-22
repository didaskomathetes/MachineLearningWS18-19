import numpy as np

from utils.data import getIrisDataAnLabels


def main():
    for n in range(1, 10):
        print(get_eig(n)[0])
    X, y = getIrisDataAnLabels("data/iris.data")
    C = np.dot(X, X.transpose())
    print("Covariance Matrix of Iris Data:")
    print(C)

    a, b = np.linalg.eig(C)
    print("Eigenvalues of Covariance Matrix of Iris Data:")
    print(a)
    print("Eigenvectors of Covariance Matrix of Iris Data:")
    print(b)


def get_eig(n):
    matrix = - 1/n * np.ones((n, n)) + np.eye(n)

    return np.linalg.eig(matrix)

