#@TODO ajout classe 
# -*- coding: utf-8 -*-

from models.classifier import Classifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import sys


class Mlp(Classifier):
    def __init__(self, hyperparams, x_train, y_train):
        super().__init__(hyperparams)
        self.x_train = x_train
        self.y_train = y_train
        self.mlp = MLPClassifier() #best params

    def set_mlp(self):
        # Permet de définir la fonction d'activation de la dernière couche à un softmax, sinon marche pas
        # Manque l'initialisation des poids mais me semble que c'est la xavier init de base :)
        # self.mlp.out_activation_ = 'softmax' fait de base qd multi-classe classification
        # Permet de set à nouveau le mlp, sinon fonctionne pas pcq existe pas
        # self.mlp = MLPClassifier(hidden_layer_sizes=self.hyperparams[0], alpha=self.hyperparams[1],
        #                        momentum=self.hyperparams[2], solver='adam', learning_rate='adaptative')
        self.mlp = MLPClassifier(solver='adam', learning_rate='adaptive')

    def train(self, training_set, target_set):
        """
        """
        print('multi-layer perceptron sklearn')
        self.set_mlp()
        # Récupérer les hyperparamètres depuis hyperparams après cross-validation
        self.mlp.fit(self.x_train, self.y_train)

    def predict(self, x):
        """
        """
        prediction_proba = self.mlp.predict_proba(x)
        if prediction_proba < 0.5:
            prediction = 0
        else:
            prediction = 1
        return prediction


    def error(self):
        """
        return actual error of the model
        """
        error = self.mlp.loss_
        return error

    def accuracy(self, X, y):
        """
        returns float,mean accuracy between dataset and labels
        """
        accuracy = self.mlp.score(X, y)
        return accuracy

if __name__ == '__main__':
    """
    Liste de fonctions mises en place pour tester certaines partie de la classe présentée précédemment. 
    # Non utile pour le bon fonctionnement du projet. 
    """

    def test_functions():
        x_normal = np.load('../Data/x_train_normal.npy')
        label_normal = np.zeros(x_normal.shape[0])
        x_pneumonia = np.load('../Data/x_train_pneumonia.npy')
        label_pneumonia = np.ones(x_pneumonia.shape[0])
        x_train = np.concatenate([x_normal, x_pneumonia])
        y_train = np.concatenate([label_normal, label_pneumonia])

        x_normal_test = np.load('../Data/x_test_normal.npy')
        x_normal_test_labels = np.zeros(x_normal_test.shape[0])
        x_pneumonia_test = np.load('../Data/x_test_pneumonia.npy')
        x_pneumonia_test_labels = np.ones(x_pneumonia_test.shape[0])
        x_test = np.concatenate([x_normal_test, x_pneumonia_test])
        y_test = np.concatenate([x_normal_test_labels, x_pneumonia_test_labels])


        hyperparams = [(2, 10, 30, 60, 100), (0.00001, 0.0001, 0.001, 0.1), (0.9, 0.7, 0.5, 0.2)]
        mlp = Mlp(hyperparams, x_train, y_train)
        mlp.train(x_train, y_train)
        print("Error : ", mlp.error())
        print("Accuracy : ", mlp.accuracy(x_test, y_test))


    test_functions()
