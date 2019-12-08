from models.classifier import Classifier
from sklearn.linear_model import SGDClassifier


class Logistic(Classifier):

    def __init__(self, hyperparams):

        # hyperparams should be a vector of : [lr, alpha]
        super().__init__(hyperparams)
        # Construction of logistic_classifier in __init__() in order to use it without training if needed:
        self.logistic_classifier = SGDClassifier(loss='log', penalty='l2', alpha=hyperparams[1],  l1_ratio=0,
                                                 fit_intercept=True, max_iter=1000, tol=None, shuffle=False,
                                                 verbose=1, n_jobs=None, random_state=None,
                                                 learning_rate='invscaling', eta0=hyperparams[0], power_t=0.5,
                                                 early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                                                 class_weight=None, warm_start=False, average=False)
        # Possibilité de passer lr à optimal ET class_weight à 'balanced'
        return

    def train(self, x_train, y_train, x_test, y_test):
        # if self.best_params is not None:
        #    self.hyperparams = self.best_params
        self.logistic_classifier.fit(x_train, y_train)
        score = self.logistic_classifier.score(x_test, y_test)
        return score

