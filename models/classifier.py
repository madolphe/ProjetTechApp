import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


class Classifier:
    """
        Parent class of all solutions of the project. Some method have to be override such as train , predict, error,
        and error_pred. Cross-validation method can be used for every solutions.
    """

    def __init__(self, hyperparams, index):
        """
        :param hyperparams: liste d'hyperparamètres
        """
        self.hyperparams = hyperparams
        self.best_params = None
        self.index = index
        pass

    def __str__(self):
        res = "Les hyperamètres sont: \n"
        for i in range(len(self.index)):
            res += self.index[i]+":"+str(self.hyperparams[i])+"\n"
        return res

    def train(self, training_set, target_set, *args):
        raise NotImplementedError

    def predict(self, x, *args):
        raise NotImplementedError

    def error(self, *args):
        raise NotImplementedError

    def error_pred(self, training_set, target_set, *args):
        raise NotImplementedError

    def probabilities(self, x_test, *args):
        raise NotImplementedError

    def cross_validation(self, ranges, training_set, target_set, k, ratio_validation, couple=[], best_params=[],
                         min_error=100000):
        """
        Recursive function used to test every permutations of ranges of parameters and test them on a subset of training
        set. Number of different hyper-parameters is not set so that this method could be used in every solutions.
        :param ranges:
        :param training_set:
        :param target_set:
        :param k:
        :param ratio_validation:
        :param couple:
        :param min_error:
        :return:
        """
        if not ranges == []:
            for elt in ranges[-1]:
                couple.append(elt)
                couple, min_error = self.cross_validation(ranges[:-1], training_set, target_set, k, ratio_validation,
                                                          best_params=best_params, couple=couple, min_error=min_error)
            return couple[:-1], min_error
        else:
            # Si l'élément L passée en paramètre est vide, c'est que nous avons fini de parcourir tous les ensembles
            print("Tuple testé:", couple)
            couple.reverse()
            self.hyperparams = couple
            taille_validation = round(ratio_validation*len(training_set))
            error = 0
            for fold in range(k):
                D = np.concatenate((training_set, target_set), axis=1)
                np.random.shuffle(D)
                (x_train, y_train) = (D[taille_validation:, :-1], D[taille_validation:, D.shape[1] - 1])
                (x_val, y_val) = (D[0:taille_validation, :-1], D[0:taille_validation, D.shape[1] - 1])
                self.train(x_train, y_train)
                error += self.error(x_val, y_val)
            print("Error moyenne: ", (error/k))
            if (error/k) < min_error:
                print(f"Tuple meilleur! On change pour:{couple} et une erreur moyenne de {error/k}")
                min_error = (error/k)
                self.best_params = self.hyperparams
            # On retourne le couple vidé de son dernier élément
            # Afin de continuer à créer des permutations
            return couple[:-1], min_error

    def get_confusion_matrix(self, x_test, y_test, verbose=True):
        """
        Method that returns confusion matrix over the test set.
        :param x_test:
        :param y_test:
        :param verbose:
        :return:
        """
        nb_normal = np.sum(y_test == 0)
        nb_pneumo = np.sum(y_test == 1)
        if verbose:
            print(f"Nombres de valeurs de test:{nb_normal}")
            print(f"Nombres d'échantillons malades:{nb_pneumo}")
            print(f"Nombres d'échantillons sains:{len(x_test)}")
        y_pred = self.predict(x_test)
        justesse = np.sum(y_pred == y_test) / len(y_test)
        print(f"Justesse: {justesse}%")
        return confusion_matrix(y_test, y_pred, normalize='all')

    def get_curves(self, x, y, plot=True):
        """
        Method that returns ROC and precision-recall curves over a set of examples.
        :param x:
        :param y:
        :param plot:
        :return:
        """
        scores = self.probabilities(x)
        roc = roc_curve(y, scores)
        pres_rec = precision_recall_curve(y, scores)
        if plot:
            plt.title("Courbe ROC")
            plt.subplot(1, 2, 1)
            plt.plot(roc[0], roc[1])
            plt.subplot(1, 2, 2)
            plt.plot(pres_rec[0], pres_rec[1])
            plt.show()
        return roc, pres_rec


if __name__ == '__main__':
    def permutation(ranges, couple=[]):
        """
        Recursive function that returns every permutations from multiple set. Used to test CV method.
        :param ranges:
        :param couple:
        :return:
        """
        if not ranges == []:
            for elt in ranges[-1]:
                # print("elt courrant:", elt)
                couple.append(elt)
                couple, _ = permutation(ranges[:-1], couple=couple)
                # print("Couple maj", couple)
            return couple[:-1]
        else:
            # Si l'élément L passée en paramètre est vide, c'est que nous avons fini de parcourir tous les ensembles
            print("Tuple testé:", couple)
            # On retourne le couple vidé de son dernier élément
            # Afin de continuer à créer des permutations
            return couple[:-1]
    # permutation(ranges=[(1, 2, 3, 4)])
