import csv
import os

import numpy as np


# loads datapoints from a file into an tuple (x,y) where x and y are arrays/vectors
def getDatapoints(filestring, type=0):
    points = []
    x = []
    y = []
    path = os.path.abspath(filestring)
    with open(path) as csv_file:
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


def getLabels(filestring):
    path = os.path.abspath(filestring)
    labels = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            labels.append(row[0])
            line_count += 1
    print(f'Processed {line_count} lines of {filestring}.')
    return np.array(labels)
