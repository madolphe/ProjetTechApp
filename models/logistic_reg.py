from models.classifier import Classifier
from sklearn.linear_model import LogisticRegression


class Logistic(Classifier):

    def __init__(self, hyperparams):
        # hyperparams should be a vector of : [lr, lambda]
        super.__init__(hyperparams)
        self.logistic_classifier = LogisticRegression(penalty='l2', dual=False, tol=0.0001, fit_intercept=True,
                                                      intercept_scaling=1, class_weight=None, random_state=None,
                                                      solver='lbfgs', max_iter=100, multi_class='auto',
                                                      verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
        return

    def train(self, training_set, target_set):
        return

