from models.classifier import Classifier
from models.svm import Svm
from models.logistic_reg import Logistic
from models.decision_tree import Forest
from models.adaboost import Adaboost
from models.mlp import Mlp
import numpy as np
from get_data.get_dataset import *
from sklearn.utils import resample
import random as random


class MixedSamples(Classifier):
    def __init__(self, index,hyperparams):
        super().__init__(index,hyperparams)
        self.svm = Svm('rbf',hyperparams=[])
        self.decision_tree = Forest([10, 3, 5, 200])
        self.adaboost  = Adaboost([0.1])
        self.mlp = Mlp([10, 0.1, 0.1])
        self.logreg = Logistic([0.01, 0.001])



    def train(self, training_set, target_set, *args):
        pass

    def predict(self, x, *args):
        pass

    def error(self, *args):
        pass

    def error_pred(self, training_set, target_set, *args):
        pass

    def probabilities(self, x_test, *args):
        pass

    def boostrap(self,training):
        print(training.shape)
        new_x_train = np.array([training[random.randint(0, len(training)-1),:] for i in range(len(training))])
        return new_x_train

    def boostrapTarget(self,target):
        print(target.shape)
        new_x_train = np.array([target[random.randint(0, len(target)-1)] for i in range(len(target))])
        return new_x_train



if __name__ == '__main__':
    x_train = training_set.loc[:, training_set.columns != 'class'].to_numpy()
    y_train = np.squeeze(training_set.loc[:, training_set.columns == 'class'].to_numpy())
    x_test = test_set.loc[:, test_set.columns != 'class'].to_numpy()
    y_test = np.squeeze(test_set.loc[:, test_set.columns == 'class'].to_numpy())


    M = MixedSamples('erer',hyperparams=[])
    M.boostrap(x_train)
