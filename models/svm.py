

from models.classifier import Classifier
from get_data.get_dataset import *
import os
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix


class Svm(Classifier):

    def __init__(self, hyperparams):
        super().__init__(hyperparams)

    def train(self, train_set, target_set):
        model = SVC(kernel='rbf')
        model.fit(train_set, target_set)
        return model

    def predict(self):
        y_pred = modelsvm.predict(X_test)
        return y_pred

    def error(self):
        pass


if __name__ == '__main__':
    X_train = training_set.drop('class',axis=1)
    Y_train = training_set['class']
    Y_train = np.where(Y_train > 0, 1, Y_train)
    Y_train = np.where(Y_train < 0, 0, Y_train)

    X_test = test_set.drop('class', axis=1)
    Y_test = test_set['class']
    Y_test = np.where(Y_test > 0, 1, Y_test)
    Y_test = np.where(Y_test < 0, 0, Y_test)

    print(Y_train)

    svm = Svm(hyperparams=0)
    modelsvm = svm.train(X_train, Y_train)
    prediction = svm.predict()
    print(confusion_matrix(Y_test, prediction))
    print(classification_report(Y_test, prediction))



