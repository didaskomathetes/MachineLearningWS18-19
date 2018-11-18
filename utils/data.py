import csv
import os

import numpy as np


# loads datapoints from a file into an tuple (x,y) where x and y are arrays/vectors
def getDatapoints(filestring, type=0, startIndex=0, endIndex=1000, array=0):
    points = []
    x = []
    y = []
    path = os.path.abspath(filestring)
    addedLabels = 0
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if (type != 0):
                points.insert(line_count, [float(row[0]), float(row[1])])
            else:
                if (line_count >= startIndex and line_count <= endIndex):
                    x.insert(line_count, float(row[0]))
                    y.insert(line_count, float(row[1]))
                    addedLabels += 1
            line_count += 1

        print(f'Processed {line_count} lines of {filestring}.  Extracted {addedLabels} datapoints')
        if (type != 0):
            return np.array(points)
        if (array == 1):
            X = []
            X.insert(0, x)
            X.insert(1, y)
            return np.array(X).transpose()
    return (np.array(x), np.array(y))


def getLabels(filestring, startIndex=0, endIndex=1000):
    path = os.path.abspath(filestring)
    labels = []
    addedLabels = 0
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if (line_count >= startIndex and line_count <= endIndex):
                labels.append(float(row[0]))
                addedLabels += 1
            line_count += 1
    print(f'Processed {line_count} lines of {filestring}. Extracted {addedLabels} data labels')
    return np.array(labels)
