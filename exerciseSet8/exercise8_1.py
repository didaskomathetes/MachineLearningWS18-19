import matplotlib.pyplot as plt
import numpy as np


def main_old():
    X=constructDataMatrix()
    print(f"DataT:\n{X}")
    C = np.dot(X, X.transpose())
    print(f"\nCovariance Matrix of river flow Data:\n{C}")

    a, b = np.linalg.eig(C)
    print(f"\nEigenvalues of Covariance Matrix of River flow Data:\n{a}")
    print(f"\nEigenvectors of Covariance Matrix of River flow Data:\n{b}")

    fig = plt.figure()
    V  = np.array([X[0], X[1]])
    print(f"\nDatapoints by PCs, column 1 (EV:{a[0]}) and 2 (EV:{a[1]}): \n{V.transpose()}\n")
    ax = fig.add_subplot(221, title="2d Graph of col 1 and 2 OLD")
    ax.scatter(X[0],X[1])

    ax = fig.add_subplot(222, projection='3d', title="3d Graph of col 1,2 and 4 OLD")
    ax.scatter(X[0],X[1],X[3])
    plt.show()


def main():
    X = constructDataMatrix()
    print(f"DataT:\n{X}")
    C = np.dot(X, X.transpose())
    print(f"\nCovariance Matrix of river flow Data:\n{C}")

    a, b = np.linalg.eig(C)
    print(f"\nEigenvalues of Covariance Matrix of River flow Data:\n{a}")
    print(f"\nEigenvectors of Covariance Matrix of River flow Data:\n{b}")

    fig = plt.figure()
    V = np.array([X[0], X[1]])
    print(f"\nDatapoints by PCs, column 1 (EV:{a[0]}) and 2 (EV:{a[1]}): \n{V.transpose()}\n")
    ax = fig.add_subplot(221, title="2d Graph of col 1 and 2")
    ax.scatter(X[0], X[1])

    ax = fig.add_subplot(222, projection='3d', title="3d Graph of col 1,2 and 4")
    ax.scatter(X[0], X[1], X[3])
    plt.show()


def constructDataMatrix():
    X=[]
    X.append([210,209,-116,-249,174,190])
    X.append([277,89,-105,-189,79,82])
    X.append([-66,122,-13,-59,95,107])
    X.append([486,296,-219,-439,253,273])
    return np.array(X)
