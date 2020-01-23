from models.classifier import Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np


class Forest(Classifier):

    def __init__(self, hyperparams):
        index = ['max_depth', 'min_samples_leaf', 'min_samples_split', 'n_estimators']
        super().__init__(hyperparams, index)
        self.forest = RandomForestClassifier()

    def train(self, training_set, target_set, tuning=False):
        if tuning:
            ranges = [(10, 20), (2, 3), (2, 5), (200, 400)]
            _ = self.cross_validation(ranges, training_set, np.expand_dims(target_set, axis=1), k=2,
                                      ratio_validation=0.1)
            self.hyperparams = self.best_params

        self.reinit()
        self.forest.fit(training_set, target_set)

    def predict(self, x, *args):
        rf_predictions = self.forest.predict(x)
        return rf_predictions

    def error_pred(self, training_set, target_set, *args):
        print(classification_report(training_set, target_set))

    def error(self, x, y):
        return np.mean(np.power(y - self.forest.predict(x), 2))

    def reinit(self):
        self.forest = RandomForestClassifier(bootstrap='True',
                                             max_depth=self.hyperparams[3],
                                             max_features='sqrt',
                                             min_samples_leaf=self.hyperparams[2],
                                             min_samples_split=self.hyperparams[1],
                                             n_estimators=self.hyperparams[0])

    def probabilities(self, x, *args):
        return self.forest.predict_proba(x)[:,1]

