import csv

import numpy as np


def main_test():
    print("Output Matrix:")
    print(X(np.array([3, 2, 4, 8]), 3))

    print(getDatapoints('data/TrainingSet1D.csv'))


def getDatapoints(filestring):
    points = []
    with open(filestring) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            points.insert(line_count, (float(row[0]), float(row[1])))
            line_count += 1
        print(f'Processed {line_count} lines of {filestring}.')
    return points


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
