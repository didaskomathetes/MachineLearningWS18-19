import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
import mpl_toolkits.mplot3d.axes3d

# followed tutorial: https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/
def main():
    fig = plt.figure()
    exercise1(fig)
    exercise2(fig)
    plt.show()


def exercise1(fig):
    X = constructDataMatrix()
    print("\n-----EXERCISE 1----")
    print(f"Data:\n{X}")
    mean = np.mean(X.transpose(), axis=1)
    print(f"Mean of X:\n{mean}")
    C = X - mean
    print(f"Data with mean removed:\n{C}")
    V = np.cov(C.transpose())
    eigVal, eigVec = np.linalg.eig(V)
    print(f"eigenValues:\n{eigVal}\nEigenVectors:\n{eigVec}")
    princComps = eigVec[:, [0, 1]]
    print(f"Principal Components:\n{princComps}")
    P = np.dot(princComps.transpose(), C.transpose())
    print(f"projected subspace:\n{P.transpose()}")

    ax = fig.add_subplot(321, title="PCA on river flow: 2d Graph of col1 and 2")
    ax.scatter(P[0], P[1])


# tutorials used: https://scikit-learn.org/stable/auto_examples/manifold/plot_swissroll.html
def exercise2(fig):
    print("\n-----EXERCISE 2----")
    X, color = make_swiss_roll(800, random_state=1234)
    ax = fig.add_subplot(322, projection='3d', title="plot of 3d swiss roll")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, s=2)
    print(f"swiss roll data shape:{X.shape}\n")
    pca = PCA(2)
    pca.fit(X)
    print(f"Principle Components:\n{pca.components_}")
    print(f"eigenvalues of principal components: { pca.explained_variance_}\n")
    P = pca.transform(X)
    print(f"shape of Projected data:{P.shape}")

    ax = fig.add_subplot(323, title="PCA on swiss roll with d=2")
    ax.scatter(P[:, 0], P[:, 1], c=color, s=2)

def constructDataMatrix():
    X = []
    X.append([210, 209, -116, -249, 174, 190])
    X.append([277, 89, -105, -189, 79, 82])
    X.append([-66, 122, -13, -59, 95, 107])
    X.append([486, 296, -219, -439, 253, 273])
    return np.array(X).transpose()


# false first try, just ignore
def main_old():
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
    ax = fig.add_subplot(221, title="2d Graph of col 1 and 2 OLD")
    ax.scatter(X[0], X[1])

    ax = fig.add_subplot(222, projection='3d', title="3d Graph of col 1,2 and 4 OLD")
    ax.scatter(X[0], X[1], X[3])
    plt.show()
