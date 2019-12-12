from models.classifier import Classifier
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import hinge_loss
from sklearn.metrics import classification_report


class Svm(Classifier):
    def __init__(self, kernel,index, hyperparams):
        """
        Constructor of the class. Added kernel to specify svm kernel we will use
        :param kernel:
        :param hyperparams:
        """
        super().__init__(hyperparams,index)
        # added kernel because of the svm implementation
        self.kernel = kernel
        self.model = SVC(kernel=kernel)

    def train(self, train_set, target_set, tuning=False):
        """
        Training function for the svm model
        :param train_set:
        :param target_set:
        :param tuning:
        :return:
        """
        # create the model with specified parameters for cross-validation:
        if self.kernel == 'rbf' and tuning:
                ranges = [(1, 10, 100, 1000), (0.001, 0.0001)]
                _ = self.cross_validation(ranges, train_set, np.expand_dims(target_set, axis=1), k=2,
                                          ratio_validation=0.1)
                self.hyperparams = self.best_params

        if self.kernel == 'poly' and tuning:
                ranges = [(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)]
                _ = self.cross_validation(ranges, train_set, np.expand_dims(target_set, axis=1), k=2,
                                          ratio_validation=0.1)
                self.hyperparams = self.best_params

        self.reinit()
        # Actually train the model with the training set and the target set
        self.model.fit(train_set, target_set)

    def predict(self, x, *args):
        # prediction by using the trained model and the target set of the testing data
        return self.model.predict(x)

    def error(self, x, y):
        """
        return actual error of the model
        """
        error = hinge_loss(y, self.model.predict(x))
        return error

    # print stats on the classification
    def error_pred(self, training_set, target_set, *args):
        print(classification_report(training_set, target_set))

    def probabilities(self, x_test, *args):
        return self.model.decision_function(x_test)

    def reinit(self):
        if self.kernel == 'rbf':
            self.model = SVC(kernel=self.kernel, gamma=self.hyperparams[1], C=self.hyperparams[0])
        elif self.kernel == 'linear':
            self.model = SVC(kernel=self.kernel)
        else:
            self.model = SVC(kernel=self.kernel, degree=self.hyperparams[0])




