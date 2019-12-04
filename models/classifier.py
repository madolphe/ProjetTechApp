import numpy as np


class Classifier:
    """
        Interface...
    """

    def __init__(self, hyperparams):
        """
        :param hyperparams: liste d'hyperparamètres
        """
        self.hyperparams = hyperparams
        pass

    def train(self, training_set, target_set):
        pass

    def predict(self, x):
        pass

    def error(self):
        pass

    def error_pred(self, training_set, target_set):
        return np.mean(training_set-target_set)

    def verbose(self):
        print("Les hyperamètres sont: ")
        print(self.hyperparams)

    def cross_validation(self, ranges, training_set, target_set, k, ratio_validation, couple=[], best_params=[],
                         min_error=0):
        """
        Fonction récursive permettant de sortir toutes les permutations d'ensembles d'hyperparamètres et les tester
        sur un jeu de validation. Méthode implémentée sans limite de nombre d'hyperparamètres.
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
                # print("elt courrant:", elt)
                couple.append(elt)
                couple = self.cross_validation(ranges[:-1], training_set, target_set, k, ratio_validation,
                                               best_params=best_params, couple=couple, min_error=min_error)
                # print("Couple maj", couple)
            return couple[:-1]
        else:
            # Si l'élément L passée en paramètre est vide, c'est que nous avons fini de parcourir tous les ensembles
            print("Tuple testé:", couple)
            self.hyperparams = couple
            taille_validation = round(ratio_validation*len(training_set))
            for fold in range(k):
                D = np.concatenate((training_set, target_set), axis=1)
                np.random.shuffle(D)
                (training_set, target_set) = (D[taille_validation:, :-1], D[taille_validation:, D.shape[1] - 1])
                (training_val, target_val) = (D[0:taille_validation, :-1], D[0:taille_validation, D.shape[1] - 1])
                self.train(training_set, training_val)
                error = self.error_pred(training_val, target_val)
                if error < min_error:
                    min_error = error
                    best_params = self.hyperparams
            # On retourne le couple vidé de son dernier élément
            # Afin de continuer à créer des permutations
            return couple[:-1]


if __name__ == '__main__':

    def permutation(ranges, couple=[]):
        """
        Fonction récursive donnant toutes les permutations d'une liste de tuples
        :param ranges:
        :param couple:
        :return:
        """
        if not ranges == []:
            for elt in ranges[-1]:
                # print("elt courrant:", elt)
                couple.append(elt)
                couple = permutation(ranges[:-1], couple=couple)
                # print("Couple maj", couple)
            return couple[:-1]
        else:
            # Si l'élément L passée en paramètre est vide, c'est que nous avons fini de parcourir tous les ensembles
            print("Tuple testé:", couple)
            # On retourne le couple vidé de son dernier élément
            # Afin de continuer à créer des permutations
            return couple[:-1]
    permutation(ranges=[(1, 2, 3, 4), (1, 2, 3, 5, 6), (1, 2, 3)])
