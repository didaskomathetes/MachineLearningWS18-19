import matplotlib.pyplot as plot
import numpy as np

from utils.data import getIrisDataAnLabels


def main():
    print("Eigenvalues of H_n for n=1...10")
    for n in range(1, 10):
        print(get_eig(n)[0])

    X, y = getIrisDataAnLabels("data/iris.data")
    C = np.dot(X, X.transpose())
    print(f"\nCovariance Matrix of Iris Data:\n{C}")

    a, b = np.linalg.eig(C)
    print(f"\nEigenvalues of Covariance Matrix of Iris Data:\n{a}")
    print(f"\nEigenvectors of Covariance Matrix of Iris Data:\n{b}")

    V = np.array([X[0], X[3]])
    print(f"\nDatapoints by PCs, column 1 (EV:{a[0]}) and 4 (EV:{a[3]}): \n{V.transpose()}\n")

    colors = []
    for kuchen in y:
        if kuchen == "Iris-setosa":
            colors.append("red")
        if kuchen == "Iris-versicolor":
            colors.append("green")
        if kuchen == "Iris-virginica":
            colors.append("blue")
    plot.scatter(X[0], X[1], c=colors)
    plot.xlabel("sepal length in cm")
    plot.ylabel("petal width in cm")
    plot.show()


def get_eig(n):
    matrix = - 1/n * np.ones((n, n)) + np.eye(n)

    return np.linalg.eig(matrix)

