import numpy as np
import random
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
    # array for means
    means = []

    for i in range(0, k):
        mean_i = random.choice(X)
        print(f"random mean from data: {mean_i}")
        while contains(means, mean_i):
            print(f"random mean from data: {mean_i} already in means")
            mean_i = random.choice(X)
        means.append(mean_i)

    print(f"means:{np.array(means)}")
    old_cluster1, old_cluster2, old_cluster3 = [], [], []
    n_cluster1, n_cluster2, n_cluster3 = calculateClusters(X, means)
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

    print(f"clusters and means after fitting in {iterations} iterations:")
    print("n_cluster1")
    print(n_cluster1)
    print("n_cluster2")
    print(n_cluster2)
    print("n_cluster3")
    print(n_cluster3)
    print(f"updated means:\n{means}")


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


def calculateMeans(n_cluster1, n_cluster2, n_cluster3):
    mean_1 = np.mean(n_cluster1, axis=0)
    mean_2 = np.mean(n_cluster2, axis=0)
    mean_3 = np.mean(n_cluster3, axis=0)
    print(f"mean_1:\n{mean_1}")
    print(f"mean_2:\n{mean_2}")
    print(f"mean_3:\n{mean_3}")
    return np.array([mean_1, mean_2, mean_3])

def contains(container, object):
    # print(f"container:{container}")
    for c_object in container:
        # print(f"c_object:{c_object}")
        for i in range(len(c_object)):
            if c_object[i] == object[i]:
                return True

    return False
