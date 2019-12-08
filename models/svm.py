

from models.classifier import Classifier
from get_data.get_dataset import *
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


class Svm(Classifier):

    # constructor of the class. Added kernel to specify svm kernel we will use
    def __init__(self, kernel, hyperparams):
        super().__init__(hyperparams)
        # added kernel because of the svm implementation
        self.kernel = kernel

    # the training function for the svm model
    def train(self, train_set, target_set):
        # create the model with a specified kernel
        if self.kernel == 'rbf':
            model = SVC(kernel=self.kernel,gamma=self.hyperparams, C=self.hyperparams)
        if self.kernel == 'linear':
            model = SVC(kernel=self.kernel)
        if self.kernel == 'poly':
            model = SVC(kernel=self.kernel, degree=self.hyperparams)
        # actually train the model with the training set and the target set
        model.fit(train_set, target_set)
        return model

    # the prediction function for the svm model
    def predictsvm(self, svmmodel, target_set):
        # prediction by using the trained model and the target set of the testing data
        y_pred = svmmodel.predict(target_set)
        return y_pred

    # print a confusion matrix and stats on the classification
    def error(self, y_pred, y_test):
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
    svmobject = Svm('rbf', hyperparams=0.5)
    # doing the training of the dataset
    modelsvm = svmobject.train(X_train, Y_train)
    # doing the prediction of the dataset
    prediction = svmobject.predictsvm(modelsvm, X_test)
    # printing statistical results of the dataset
    svmobject.error(prediction, Y_test)




