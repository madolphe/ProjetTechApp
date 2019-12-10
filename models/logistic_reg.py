from models.classifier import Classifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
import numpy as np


class Logistic(Classifier):
    def __init__(self, hyperparams):
        index = ['lr', 'alpha']
        super().__init__(hyperparams, index)
        # Construction of logistic_classifier in __init__() in order to use it without training if needed:
        self.logistic_classifier = SGDClassifier(loss='log', penalty='l2', alpha=hyperparams[1],  l1_ratio=0,
                                                 fit_intercept=True, max_iter=1000, tol=None, shuffle=False,
                                                 verbose=0, n_jobs=None, random_state=None,
                                                 learning_rate='invscaling', eta0=hyperparams[0], power_t=0.5,
                                                 early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                                                 class_weight=None, warm_start=False, average=False)

    def train(self, training_set, target_set, tuning=False):
        """
        Train method with optional cross-validation. If CV is set to True, self.logistic_classifier will be finally
        trained with best_parameters found during CV.
        :param training_set:
        :param target_set:
        :param tuning:
        """
        if tuning:
            # For cross validation:
            ranges = [(0.0001, 0.001, 0.01, 0.1), (0.1, 0.2, 0.5, 1, 2)]
            self.cross_validation(ranges, training_set, np.expand_dims(target_set, axis=1), k=5, ratio_validation=0.1)
            # When cross_val is done, we update our parameters:
            self.hyperparams = self.best_params
        # Each time, train is called, we re-init a SGDClassifier object so that we could use correct parameters:
        self.reinit()
        self.logistic_classifier = self.logistic_classifier.fit(training_set, target_set)
        return

    def reinit(self):
        """
        Re-init model without any "fit" history. Useful if user wants to set new parameters stored in self.hyperparams
        :return:
        """
        self.logistic_classifier = SGDClassifier(loss='log', penalty='l2', alpha=self.hyperparams[1], l1_ratio=0,
                                                 fit_intercept=True, max_iter=1000, tol=None, shuffle=False,
                                                 verbose=0, n_jobs=None, random_state=None,
                                                 learning_rate='invscaling', eta0=self.hyperparams[0], power_t=0.5,
                                                 early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                                                 class_weight=None, warm_start=False, average=False)

    def error(self, x, y):
        return log_loss(y, self.logistic_classifier.predict(x))

    def predict(self, x, *args):
        return self.logistic_classifier.predict(x)

    def probabilities(self, x, *args):
        return self.logistic_classifier.decision_function(x)

    def error_pred(self, training_set, target_set, *args):
        pass

