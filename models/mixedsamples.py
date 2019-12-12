from models.classifier import Classifier
import numpy as np
from get_data.get_dataset import *
from sklearn.utils import resample


class MixedSamples(Classifier):
    def __init__(self, index,hyperparams):
        super().__init__(index,hyperparams)

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

    def bootstrap(self,training):
        print(training[:,0])
        pass


if __name__ == '__main__':
    x_train = training_set.loc[:, training_set.columns != 'class'].to_numpy()
    y_train = np.squeeze(training_set.loc[:, training_set.columns == 'class'].to_numpy())
    x_test = test_set.loc[:, test_set.columns != 'class'].to_numpy()
    y_test = np.squeeze(test_set.loc[:, test_set.columns == 'class'].to_numpy())

    print(x_train)
    #print(x_train.shape)

    M = MixedSamples('erer',hyperparams=[])
    M.bootstrap(x_test)