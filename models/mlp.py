#@TODO ajout classe 
# -*- coding: utf-8 -*-

from .classifier import Classifier
from sklearn.neural_network import MLPClassifier


class Mlp(Classifier):
    def __init__(self, hyperparams, x_train, y_train, x_val, y_val):
        super().__init__(hyperparams)
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.mlp = MLPClassifier() #best params

    def set_mlp(self):
        # Permet de définir la fonction d'activation de la dernière couche à un softmax, sinon marche pas
        # Manque l'initialisation des poids mais me semble que c'est la xavier init de base :)
        self.mlp.out_activation_ = 'softmax'
        # Permet de set à nouveau le mlp, sinon fonctionne pas pcq existe pas
        self.mlp = MLPClassifier(hidden_layer_sizes=self.hyperparams[0], alpha=self.hyperparams[1],
                                 momentum=self.hyperparams[2], solver='adam', learning_rate='adaptative')


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
        #@TODO modifier la structure de l'interface pour qu'il prenne bien une entrée x
        X = 0 # A modifier dans la structure de l'interface :)
        prediction = self.mlp.predict(X)
        return prediction


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
