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

    for i in range(1, k + 1):
        mean_i = random.choice(X)
        print(f"random mean from data: {mean_i}")
        while contains(means, mean_i):
            print(f"random mean from data: {mean_i} already in means")
            mean_i = random.choice(X)
        means.append(mean_i)

    print(f"means:{np.array(means)}")


def contains(container, object):
    # print(f"container:{container}")
    for c_object in container:
        # print(f"c_object:{c_object}")
        for i in range(len(c_object)):
            if c_object[i] == object[i]:
                return True

    return False
