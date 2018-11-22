import numpy as np

#from utils.data import getDatapoints

def main():
    for n in range(1, 10):
        print(get_eig(n)[0])


def get_eig(n):
    matrix = -1 / n * np.ones((n, n)) + np.eye(n)

    return np.linalg.eig(matrix)

