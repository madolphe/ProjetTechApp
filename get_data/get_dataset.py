import numpy as np
import os
import pandas as pd
from sys import getsizeof

x_train_normal = np.load(os.path.join("Data", "chest_xray_features_vect", "x_train_normal.npy"))
x_train_pneumonia = np.load(os.path.join("Data", "chest_xray_features_vect", "x_train_pneumonia.npy"))
x_val_normal = np.load(os.path.join("Data", "chest_xray_features_vect", "x_val_normal.npy"))
x_val_pneumonia = np.load(os.path.join("Data", "chest_xray_features_vect", "x_val_pneumonia.npy"))
x_test_normal = np.load(os.path.join("Data", "chest_xray_features_vect", "x_test_normal.npy"))
x_test_pneumonia = np.load(os.path.join("Data", "chest_xray_features_vect", "x_test_pneumonia.npy"))

# Last column is going to be class label: 0 normal, 1 pneumonia
x_train_normal = np.insert(x_train_normal, x_train_normal.shape[1], 0, axis=1)
x_train_pneumonia = np.insert(x_train_pneumonia, x_train_pneumonia.shape[1], 1, axis=1)
x_val_normal = np.insert(x_val_normal, x_val_normal.shape[1], 0, axis=1)
x_val_pneumonia = np.insert(x_val_pneumonia, x_val_pneumonia.shape[1], 1, axis=1)
x_test_normal = np.insert(x_test_normal, x_test_normal.shape[1], 0, axis=1)
x_test_pneumonia = np.insert(x_test_pneumonia, x_test_pneumonia.shape[1], 1, axis=1)

# As cross-validation create its own validation dataset, we can concatenate val and train sets:
x_train_normal = np.concatenate((x_train_normal, x_val_normal), axis=0)
x_train_pneumonia = np.concatenate((x_train_pneumonia, x_val_pneumonia), axis=0)

# Final training and test set should have normal and pneumonia:
training_set = pd.DataFrame(np.concatenate((x_train_pneumonia, x_train_normal), axis=0))
test_set = pd.DataFrame(np.concatenate((x_test_pneumonia, x_test_normal), axis=0))

# First let's name our columns:
columns = np.array(['contrast_0', 'contrast_45', 'contrast_90', 'contrast_135',
                    'correlation_0', 'correlation_45', 'correlation_90', 'correlation_135',
                    'energy_0', 'energy_45', 'energy_90', 'energy_135',
                    'homogeneity_0', 'homogeneity_45', 'homogeneity_90', 'homogeneity_135', 'mean4LF',
                    'std4LF', 'mean3HF_H', 'std3HF_H', 'mean3HF_V', 'std3HF_V', 'mean3HF_D', 'std3HF_D',
                    'mean2HF_H', 'std2HF_H', 'mean2HF_V', 'std2HF_V', 'mean2HF_D', 'std2HF_D', 'mean1HF_H',
                    'std1HF_H', 'mean1HF_V', 'std1HF_V', 'mean1HF_D', 'std1HF_D', 'class'])
training_set.columns = columns
test_set.columns = columns

# As we figured out, some features are "useless" we could drop them
training_set = training_set.drop(['mean2HF_H', 'mean1HF_V', 'mean1HF_H', 'mean3HF_H', 'mean2HF_V', 'mean3HF_D'], axis=1)
test_set = test_set.drop(['mean2HF_H', 'mean1HF_V', 'mean1HF_H', 'mean3HF_H', 'mean2HF_V', 'mean3HF_D'], axis=1)

# To finish the preprocessing she normalize our datas:
training_set = (training_set-training_set.mean()) / training_set.std()
test_set = (test_set-test_set.mean()) / test_set.std()

if __name__ == '__main__':

    def infos():
        print(training_set.head())
        print(test_set.head())
        print("Approx of the size of our training_dataset in memory: ", ((getsizeof(training_set) / 8)/10**3), "Mo.")
        print("Approx of the size of our test_dataset in memory: ", ((getsizeof(test_set) / 8)/10**3), "Mo.")
    # infos()

