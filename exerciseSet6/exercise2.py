import matplotlib.pyplot as plt
import numpy as np

from utils.data import getDatapoints
from utils.data import getLabels


def main():
    y = getLabels('data/svm_labels.csv')

    plotDataAndDecisionBoundary(sampleBoundaryFunction)


def plotDataAndDecisionBoundary(decisionBoundaryFunction):
    plt.subplot(221)

    plt.gca().set_title("Datapoints and decision boundary")
    class1x_1, class1x_2 = getDatapoints('data/svm_data.csv', endIndex=99)
    other_classx_1, other_classx_2 = getDatapoints('data/svm_data.csv', startIndex=100)

    plt.plot(class1x_1, class1x_2, 'r+', label="class 1")
    plt.plot(other_classx_1, other_classx_2, 'b+', label="class -1")

    x = np.linspace(-2, 8, 100)
    plt.plot(x, decisionBoundaryFunction(x), ":", label='sample decision boundary')

    plt.axis([-3, 8, -3, 8])
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.grid(True)
    plt.legend()
    plt.show()


def sampleBoundaryFunction(x):
    return -((9 / 7) * x) + 5
