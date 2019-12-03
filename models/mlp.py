#@TODO ajout classe 
# -*- coding: utf-8 -*-

import numpy as np
from .classifier import Classifier
import sys
from sklearn.neural_network import MLPClassifier


class Mlp(Classifier):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.mlp = MLPClassifier() #best params

    def set_mlp(self):
        self.mlp = MLPClassifier() #new params after cross-val


    def train(self, training_set, target_set):
        """
        """
        print('multi-layer perceptron sklearn')
        self.set_mlp()
        # Récupérer les hyperparamètres depuis hyperparams après cross-validation
        self.mlp.fit(training_set, target_set)

    def predict(self):
        """
        """
        pass

    def error(self):
        """
        """
        pass

    def error_pred(self, training_set, target_set):
        """

        :param training_set:
        :param target_set:
        :return:
        """
        pass


if __name__ == '__main__':
    """
    Liste de fonctions mises en place pour tester certaines partie de la classe présentée précédemment. 
    # Non utile pour le bon fonctionnement du projet. 
    """
