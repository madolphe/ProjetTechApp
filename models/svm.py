

from models.classifier import Classifier
from get_data.get_dataset import *
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import hinge_loss
from sklearn.metrics import classification_report, confusion_matrix


class Svm(Classifier):

    # constructor of the class. Added kernel to specify svm kernel we will use
    def __init__(self, kernel, hyperparams):
        super().__init__(hyperparams)
        # added kernel because of the svm implementation
        self.kernel = kernel
        self.model = SVC(kernel=kernel)

    # the training function for the svm model
    def train(self, train_set, target_set, tuning=False):
        # create the model with a specified kernel
        if self.kernel == 'rbf':
            if tuning:
                ranges = [(1,10,100,1000), (0.001,0.0001)]
                _ = self.cross_validation(ranges, train_set, np.expand_dims(target_set, axis=1), k=2,
                                          ratio_validation=0.1)
                self.hyperparams = self.best_params
            self.model = SVC(kernel=self.kernel, gamma=self.hyperparams[1], C=self.hyperparams[0])

        if self.kernel == 'linear':
            self.model = SVC(kernel=self.kernel)

        if self.kernel == 'poly':
            if tuning:
                ranges = [(1,2,3)]
                _ = self.cross_validation(ranges, train_set, np.expand_dims(target_set, axis=1), k=5,
                                          ratio_validation=0.1)
                self.hyperparams = self.best_params
            self.model = SVC(kernel=self.kernel, degree=self.hyperparams[0])
        # actually train the model with the training set and the target set
        self.model.fit(train_set, target_set)

    # the prediction function for the svm model
    def predictsvm(self, target_set):
        # prediction by using the trained model and the target set of the testing data
        y_pred = self.model.predict(target_set)
        return y_pred

    def error(self, x, y):
        """
        return actual error of the model
        """
        error = hinge_loss(y, self.model.predict(x))
        return error

    # print a confusion matrix and stats on the classification
    def error_pred(self, y_pred, y_test):
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    # creating the train dataset by removing the data of the class
    X_train = training_set.drop('class',axis=1)
    # creating the target dataset of the train dataset with the class column
    Y_train = training_set['class']

    # change value of class to 1 if they are superior to 0
    Y_train = np.where(Y_train > 0, 1, Y_train)
    # change value of class to 0 if they are less than 0
    Y_train = np.where(Y_train < 0, 0, Y_train)

    # creating the test dataset by removing the data of the class
    X_test = test_set.drop('class', axis=1)
    # creating the target dataset of the train dataset with the class column
    Y_test = test_set['class']
    # change value of class to 1 if they are superior to 0
    Y_test = np.where(Y_test > 0, 1, Y_test)
    # change value of class to 0 if they are less than 0
    Y_test = np.where(Y_test < 0, 0, Y_test)

    # creating a svm object
    svmobject = Svm('rbf', hyperparams=[])
    # doing the training of the dataset
    svmobject.train(X_train, Y_train, tuning=True)
    # doing the prediction of the dataset
    prediction = svmobject.predictsvm(X_test)

    # printing statistical results of the dataset
    svmobject.error_pred(prediction, Y_test)




