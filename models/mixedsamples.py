from models.classifier import Classifier
from models.svm import Svm
from models.logistic_reg import Logistic
from models.decision_tree import Forest
from models.adaboost import Adaboost
from models.mlp import Mlp
from sklearn.utils import resample
import random as random
import numpy as np


class MixedSamples(Classifier):
    def __init__(self, index,hyperparams):
        super().__init__(index,hyperparams)
        self.svm = Svm('rbf',['gamma','regularisation'],[100,0.0001])
        self.decision_tree = Forest([10, 3, 5, 200])
        self.adaboost  = Adaboost([0.1])
        self.mlp = Mlp([10, 0.1, 0.1])
        self.logreg = Logistic([0.01, 0.001])

    def train(self, training_set, target_set, *args):
        self.reinit()
        x_train1,y_train1 = self.boostrap(training_set,target_set)
        x_train2, y_train2 = self.boostrap(training_set, target_set)
        x_train3, y_train3 = self.boostrap(training_set, target_set)
        x_train4, y_train4 = self.boostrap(training_set, target_set)
        x_train5, y_train5 = self.boostrap(training_set, target_set)

        self.svm.train(x_train1, y_train1, tuning=False)
        self.decision_tree.train(x_train2, y_train2, tuning=False)
        self.adaboost.train(x_train3, y_train3, tuning=False)
        self.mlp.train(x_train4, y_train4, tuning=False)
        self.logreg.train(x_train5, y_train5, tuning=False)

    def predict(self, x, *args):
        predictsvm = self.svm.predict(x)
        predictada = self.adaboost.predict(x)
        predictlog = self.logreg.predict(x)
        predicttree = self.decision_tree.predict(x)
        predictmlp = self.mlp.predict(x)

        totalpredict = np.array([predictada,predictlog,predictmlp,predictsvm,predicttree]).T
        print(totalpredict.shape)
        totalpredict = np.sum(totalpredict,axis=1)
        totalpredict = [1 if totalpredict[i] > 3 else 0 for i in range(totalpredict.shape[0])]
        return totalpredict

    def probabilities(self, x_test, *args):
        predictsvm = self.svm.predict(x_test)
        predictada = self.adaboost.predict(x_test)
        predictlog = self.logreg.predict(x_test)
        predicttree = self.decision_tree.predict(x_test)
        predictmlp = self.mlp.predict(x_test)

        totalpredict = np.array([predictada, predictlog, predictmlp, predictsvm, predicttree]).T

        return np.mean(totalpredict,axis=1)


    def reinit(self):
        """
        Re-init model without any "fit" history. Useful if user wants to set new parameters stored in self.hyperparams
        :return:
        """
        self.svm = Svm('rbf', ['gamma', 'regularisation'], [100, 0.0001])
        self.decision_tree = Forest([10, 3, 5, 200])
        self.adaboost = Adaboost([0.1])
        self.mlp = Mlp([10, 0.1, 0.1])
        self.logreg = Logistic([0.01, 0.001])




    def error(self, *args):
        pass

    def error_pred(self, training_set, target_set, *args):
        pass


    def boostrap(self,training,target):
        x_new_train = []
        y_new_train = []
        for i in range(training.shape[0]):
            index = random.randint(0, len(training)-1)
            x_new_train.append(training[index,:])
            y_new_train.append(target[index])

        #new_x_train = np.array([training[random.randint(0, len(training)-1),:] for i in range(len(training))])
        return x_new_train, y_new_train



