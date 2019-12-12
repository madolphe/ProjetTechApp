from models.classifier import Classifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np


class Adaboost(Classifier):
    def __init__(self, hyperparams):
        index = ['lr']
        super().__init__(hyperparams, index)
        # Construction of logistic_classifier in __init__() in order to use it without training if needed:
        # If base_estimator is set to None, then stamp function is used (Decision-tree with depth=1)
        self.adaboost_classifier = AdaBoostClassifier(base_estimator=None, n_estimators=50,
                                                      learning_rate=self.hyperparams[0])

    def __str__(self):
        res = "Estimator="+str(self.adaboost_classifier.base_estimator_)
        res += super().__str__()
        print(res)

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
            ranges = []
            self.cross_validation(ranges, training_set, np.expand_dims(target_set, axis=1), k=5, ratio_validation=0.1)
            # When cross_val is done, we update our parameters:
            self.hyperparams = self.best_params
        # Each time, train is called, we re-init a SGDClassifier object so that we could use correct parameters:
        self.reinit()
        self.adaboost_classifier.fit(training_set, target_set)

    def reinit(self):
        """
        Re-init model without any "fit" history. Useful if user wants to set new parameters stored in self.hyperparams
        :return:
        """
        self.adaboost_classifier = AdaBoostClassifier(base_estimator=None, n_estimators=50,
                                                      learning_rate=self.hyperparams[0])

    def error(self, x, y):
        return

    def predict(self, x, *args):
        return self.adaboost_classifier.predict(x)

    def probabilities(self, x, *args):
        return self.adaboost_classifier.decision_function(x)

    def error_pred(self, training_set, target_set, *args):
        pass

