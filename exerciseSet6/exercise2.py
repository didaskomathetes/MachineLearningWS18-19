from utils.data import getDatapoints
from utils.data import getLabels


def main():
    x_1, x_2 = getDatapoints('data/svm_data.csv')
    y = getLabels('data/svm_labels.csv')
    print(f"Datapoints: {x_1,x_2} and Labels {y} ")
