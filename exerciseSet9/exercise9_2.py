import numpy as np
import random
import matplotlib.pyplot as plt
from data import getIrisDataAnLabels


def main():
    X, y = getIrisDataAnLabels("exerciseSet7/data/iris.data")
    X = X.transpose()
    # use only petal length and width
    X = np.array([X[:, 2], X[:, 3]]).transpose()
    print("data and shape:")
    print(X)
    print(X.shape)
    # number of groups:
    k = 3
    means = []
    # assign mean randomly from datapoints
    for i in range(0, k):
        mean_i = random.choice(X)
        while contains(means, mean_i):
            print(f"random mean from data: {mean_i} already in means")
            mean_i = random.choice(X)
        means.append(mean_i)
    means = np.array(means)

    print(f"initial random means:\n{means}")
    old_cluster1, old_cluster2, old_cluster3 = [], [], []
    iterations = 0
    while True:
        iterations += 1
        n_cluster1, n_cluster2, n_cluster3 = calculateClusters(X, means)
        if np.array_equal(n_cluster1, old_cluster1) and np.array_equal(n_cluster2, old_cluster2) and np.array_equal(
                old_cluster3, n_cluster3):
            break

        means = calculateMeans(n_cluster1, n_cluster2, n_cluster3)
        old_cluster1 = n_cluster1
        old_cluster2 = n_cluster2
        old_cluster3 = n_cluster3

    print(f"clusters converged in {iterations} iterations:")
    print("n_cluster1")
    print(n_cluster1)
    print("n_cluster2")
    print(n_cluster2)
    print("n_cluster3")
    print(n_cluster3)
    print(f"updated means:\n{means}")

    fig = plt.figure()
    ax = fig.add_subplot(111, title="Iris data with k-means")
    plotData(n_cluster1, "red", means[0], ax)
    plotData(n_cluster2, "blue", means[1], ax)
    plotData(n_cluster3, "green", means[2], ax)
    plt.xlabel("Petal length in cm")
    plt.ylabel("Petal width in cm")
    plt.show()

def calculateClusters(X, means):
    cluster1 = []
    cluster2 = []
    cluster3 = []
    # calculate distances for each point and append point to corresponding cluster
    for i in range(0, X.shape[0]):
        dist1 = np.linalg.norm(X[i] - means[0])
        dist2 = np.linalg.norm(X[i] - means[1])
        dist3 = np.linalg.norm(X[i] - means[2])
        distMin = min(dist1, dist2, dist3)

        if dist1 == distMin:
            cluster1.append(X[i])
            continue
        if dist2 == distMin:
            cluster2.append(X[i])
            continue
        if dist3 == distMin:
            cluster3.append(X[i])
            continue

    return np.array(cluster1), np.array(cluster2), np.array(cluster3)


#calculate the means of each cluster
def calculateMeans(n_cluster1, n_cluster2, n_cluster3):
    mean_1 = np.mean(n_cluster1, axis=0)
    mean_2 = np.mean(n_cluster2, axis=0)
    mean_3 = np.mean(n_cluster3, axis=0)
    return np.array([mean_1, mean_2, mean_3])


def plotData(cluster, color, mean, ax):
    ax.scatter(cluster[:, 0], cluster[:, 1], c=color, s=5)
    ax.scatter(mean[0], mean[1], c=color, marker="+")


def contains(container, object):
    for c_object in container:
        for i in range(len(c_object)):
            if c_object[i] == object[i]:
                return True
    return False