

from .classifier import Classifier
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

data_dir = os.path.join('..', 'Data', 'chest_xray_features_vect')
print(data_dir)

train_normal_dir = os.path.join(data_dir, 'x_train_normal.npy')
test_normal_dir = os.path.join(data_dir, 'x_test_normal.npy')
val_normal_dir = os.path.join(data_dir, 'x_val_normal.npy')

train_pneumonia_dir = os.path.join(data_dir, 'x_train_pneumonia.npy')
test_pneumonia_dir = os.path.join(data_dir, 'x_test_pneumonia.npy')
val_pneunomia_dir = os.path.join(data_dir, 'x_val_pneumonia.npy')


class Svm(Classifier):

    def __init__(self, hyperparams):
        super().__init__(hyperparams)

    def train(self, training_set, target_set):
        pass



if __name__ == '__main__':
    data = np.load(train_normal_dir)
    data2 = np.load(train_pneumonia_dir)
    plt.scatter(data[0], data2[1], c=y, cmap='winter')
    plt.show()