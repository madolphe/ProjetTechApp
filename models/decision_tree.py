#@TODO ajout classe 
from models.classifier import Classifier
from sklearn.tree import DecisionTreeClassifier
from get_data.get_dataset import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import export_graphviz
from matplotlib import pyplot as plt
from graphviz import Source
from subprocess import call
from sklearn.metrics import hinge_loss



class Forest(Classifier):

    def __init__(self,hyperparams):
        super().__init__(hyperparams)
        self.forest = RandomForestClassifier()

    def train(self, train_set, target_set, tuning=False):
       if tuning:
            ranges = [(10, 20, 30), (1, 2, 4), (2, 5, 10), (200, 400, 600)]
            _ = self.cross_validation(ranges, train_set, np.expand_dims(target_set, axis=1), k=2,ratio_validation=0.1)
            self.hyperparams = self.best_params


       self.forest = RandomForestClassifier(bootstrap = 'True',
                                            max_depth = self.hyperparams[3],
                                            max_features = 'sqrt',
                                            min_samples_leaf = self.hyperparams[2],
                                            min_samples_split = self.hyperparams[1],
                                            n_estimators = self.hyperparams[0])
       self.forest.fit(train_set, target_set)



    def predictForest(self, target_set):
        rf_predictions = self.forest.predict(target_set)
        rf_probs = self.forest.predict_proba(target_set)[:, 1]
        return rf_predictions

    def error_pred(self, y_pred, y_test):
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    def error(self, x, y):
        """
        return actual error of the model
        """
        error = hinge_loss(y, self.forest.predict(x))
        return error

if __name__ == '__main__':
    # creating the train dataset by removing the data of the class
    X_train = training_set.drop('class',axis=1)
    # creating the target dataset of the train dataset with the class column
    Y_train = training_set['class']

    # change value of class to 1 if they are superior to 0
    Y_train = np.where(Y_train > 0, 1, Y_train)
    # change value of class to 0 if they are less than 0
    Y_train = np.where(Y_train < 0, 0, Y_train)

    # creating the test dataset by removing the data of the class
    X_test = test_set.drop('class', axis=1)
    # creating the target dataset of the train dataset with the class column
    Y_test = test_set['class']
    # change value of class to 1 if they are superior to 0
    Y_test = np.where(Y_test > 0, 1, Y_test)
    # change value of class to 0 if they are less than 0
    Y_test = np.where(Y_test < 0, 0, Y_test)

    forestobject = Forest(hyperparams=[])
    forestobject.train(X_train, Y_train,tuning=True)
    prediction = forestobject.predictForest(X_test)
    forestobject.error_pred(prediction, Y_test)


