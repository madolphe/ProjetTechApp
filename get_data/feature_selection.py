from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import pandas as pd


# First method, variance threshold:
def variance_threshold(data_set, columns):
    """
    Method used to print sorted variances of each features
    :param data_set:
    :param columns:
    :return:
    """
    print(data_set.shape)
    # First method: Variance Threshold
    # In order to compare same scales variances, we scale them in the range [0,1]:
    data_set_norm = MinMaxScaler().fit_transform(data_set[:, :-1])
    # Then we create our dataset and compute variance:
    df = pd.DataFrame(data_set_norm)
    df.columns = columns[:-1]
    var = pd.DataFrame(df.var(axis=0))
    var.columns = ['Variance']
    # We sort variances in ascending order:
    var = var.sort_values(by=['Variance'])
    print(var.head(36))
    pass


# Second method, chi-squared:
def chi_carre(data_set):
    """
    Method used to print sorted chi squarred score of each features
    :param data_set:
    :return:
    """
    # Scale our datas:
    data_set_norm = MinMaxScaler().fit_transform(data_set[:, :-1])
    chi_selector = SelectKBest(score_func=chi2, k=10)
    # get the chi-squared score:
    fit = chi_selector.fit(data_set_norm, y)
    # Push results in df and sort it:
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(columns)
    feature_score = pd.concat([dfcolumns, dfscores], axis=1)
    feature_score.columns = ['Feature', 'Score']
    print(feature_score.nlargest(36, 'Score'))
    pass


if __name__ == '__main__':

    x_train_normal = np.load(os.path.join("..", "Data", "chest_xray_features_vect", "x_train_normal.npy"))
    x_train_pneumonia = np.load(os.path.join("..", "Data", "chest_xray_features_vect", "x_train_pneumonia.npy"))
    # Last column is going to be class label: 0 normal, 1 pneumonia
    x_train_normal = np.insert(x_train_normal, x_train_normal.shape[1], 0, axis=1)
    x_train_pneumonia = np.insert(x_train_pneumonia, x_train_pneumonia.shape[1], 1, axis=1)
    data_set = np.concatenate((x_train_pneumonia, x_train_normal), axis=0)
    columns = np.array(['contrast_0', 'contrast_45', 'contrast_90', 'contrast_135',
                        'correlation_0', 'correlation_45', 'correlation_90', 'correlation_135',
                        'energy_0', 'energy_45', 'energy_90', 'energy_135',
                        'homogeneity_0', 'homogeneity_45', 'homogeneity_90', 'homogeneity_135', 'mean4LF',
                        'std4LF', 'mean3HF_H', 'std3HF_H', 'mean3HF_V', 'std3HF_V', 'mean3HF_D', 'std3HF_D',
                        'mean2HF_H', 'std2HF_H', 'mean2HF_V', 'std2HF_V', 'mean2HF_D', 'std2HF_D', 'mean1HF_H',
                        'std1HF_H', 'mean1HF_V', 'std1HF_V', 'mean1HF_D', 'std1HF_D', 'class'])
    y = data_set[:, -1]

    variance_threshold(data_set, columns)
    chi_carre(data_set)
    # With previous results, we decide to drop the following features:
    # ['mean2HF_H', 'mean1HF_V', 'mean1HF_H', 'mean3HF_H', 'mean2HF_V', 'mean3HF_D']


