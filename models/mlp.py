# -*- coding: utf-8 -*-
from models.classifier import Classifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
import numpy as np


class Mlp(Classifier):
    def __init__(self, hyperparams):
        index = ['hidden_layer_sizes', 'alpha', 'momentum']
        super().__init__(hyperparams, index)
        self.mlp = MLPClassifier(max_iter=1, hidden_layer_sizes=self.hyperparams[0], alpha=self.hyperparams[1],
                                 momentum=self.hyperparams[2], solver='adam', learning_rate='adaptive',
                                 warm_start=False)

    def train(self, training_set, target_set, tuning=False):
        """
        """
        if tuning:
            # Tested values for cross-validation:
            ranges = [(2, 10, 30, 60, 100), (0.9, 0.7, 0.5, 0.2), (0.001, 0.1, 0.5)]
            _ = self.cross_validation(ranges, training_set, np.expand_dims(target_set, axis=1), k=2,
                                      ratio_validation=0.1)
            # When cross_val is done, we update our parameters:
            self.hyperparams = self.best_params
        # Each time, train is called, we re-init a SGDClassifier object so that we could use correct parameters:
        self.reinit()
        self.mlp.fit(training_set, target_set)

    def reinit(self):
        self.mlp = MLPClassifier(max_iter=1, hidden_layer_sizes=self.hyperparams[0], alpha=self.hyperparams[1],
                                 momentum=self.hyperparams[2], solver='adam', learning_rate='adaptive',
                                 warm_start=False)

    def predict(self, x, *args):
        """
        """
        return self.mlp.predict(x)

    def error(self, x, y):
        """
        return actual error of the model
        """
        return log_loss(y, self.mlp.predict(x))

    def error_pred(self, training_set, target_set, *args):
        """
        Returns mean accuracy of prediction over a set
        """
        return self.mlp.score(training_set, target_set)

    def probabilities(self, x_test, *args):
        return self.mlp.predict_proba(x_test)[:, 1]

    def check_over_fitting(self, training_set, target_set):
        """
        Check if model is able to overfit on small dataset and many epochs
        :return:
        """
        n_check = 5
        x_check = training_set[:n_check]
        y_check = target_set[:n_check]
        self.mlp.fit(x_check, y_check)
        accuracy = self.error_pred(x_check, y_check)
        print('Accuracy d\'entraînement, devrait être 1.0: {:.3f}'.format(accuracy))
        if accuracy < 0.98:
            print('ATTENTION: L\'accuracy n\'est pas 100%.')
        else:
            print('SUCCÈS')


