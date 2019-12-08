# -*- coding: utf-8 -*-

from models.classifier import Classifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
import numpy as np


class Mlp(Classifier):
    def __init__(self, hyperparams, x_train, y_train):
        super().__init__(hyperparams)
        self.x_train = x_train
        self.y_train = y_train
        self.mlp = MLPClassifier(max_iter=1, hidden_layer_sizes=self.hyperparams[0], alpha=self.hyperparams[1],
                                 momentum=self.hyperparams[2], solver='adam', learning_rate='adaptive',
                                 warm_start=False)

    def train(self, training_set, target_set, tuning=False):
        """
        """
        if tuning:
            ranges = [(2, 10, 30, 60, 100), (0.00001, 0.0001, 0.001, 0.1), (0.9, 0.7, 0.5, 0.2)]
            _ = self.cross_validation(ranges, self.x_train, np.expand_dims(self.y_train, axis=1), k=2,
                                      ratio_validation=0.1)
            self.hyperparams = self.best_params
            print("Hyperparms choisis: ", self.hyperparams)
        self.mlp = MLPClassifier(hidden_layer_sizes=self.hyperparams[2], alpha=self.hyperparams[1],
                                 momentum=self.hyperparams[0], solver='adam', learning_rate='adaptive')
        self.mlp.fit(self.x_train, self.y_train)

    def predict(self, x):
        """
        Permet de définir la prédiction du modèle d'une donnée nouvelle une fois qu'il est bien entraîné
        """
        prediction_proba = self.mlp.predict_proba(x)
        if prediction_proba < 0.5:
            prediction = 0
        else:
            prediction = 1
        return prediction

    def error(self, x, y):
        """
        return actual error of the model
        """
        error = log_loss(y, self.mlp.predict(x))
        return error

    def error_pred(self, training_set, target_set):
        """
        returns float,mean accuracy between dataset and labels
        """
        accuracy = self.mlp.score(training_set, target_set)
        return accuracy

    def check_overfitting(self):
        """
        Permet de vérifier que le modèle puisse apprendre, en vérifiant qu'il soit capable de faire
        du sur-apprentissage
        :return:
        """
        n_check = 5
        X_check = self.x_train[:n_check]
        y_check = self.y_train[:n_check]
        self.mlp.fit(X_check, y_check)
        accuracy = self.error_pred(X_check, y_check)
        print('Accuracy d\'entraînement, devrait être 1.0: {:.3f}'.format(accuracy))
        if accuracy < 0.98:
            print('ATTENTION: L\'accuracy n\'est pas 100%.')
        else:
            print('SUCCÈS')

    def accuracy(self, x, y):
        accuracy = self.mlp.score(x, y)
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

        hyperparams = [1000, 0.001, 0.7]
        mlp = Mlp(hyperparams, x_train, y_train)
        mlp.train(x_train, y_train, True)
        print("Error on training set : ", mlp.error(x_train, y_train))
        print("Accuracy on testing set : ", mlp.accuracy(x_test, y_test))


    test_functions()
