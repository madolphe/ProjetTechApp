import os
from glob import glob

data_dir = os.path.join('..', 'Data', 'chest_xray')
print(data_dir)

train_normal_dir = os.path.join(data_dir, 'train', 'NORMAL')
test_normal_dir = os.path.join(data_dir, 'test', 'NORMAL')
val_normal_dir = os.path.join(data_dir, 'val', 'NORMAL')

train_pneumonia_dir = os.path.join(data_dir, 'train', 'PNEUMONIA')
test_pneumonia_dir = os.path.join(data_dir, 'test', 'PNEUMONIA')
val_pneunomia_dir = os.path.join(data_dir, 'val', 'PNEUMONIA')

x_train_normal = glob(os.path.join(train_normal_dir, '*.jpeg'))
x_train_pneumonia = glob(os.path.join(train_pneumonia_dir, '*.jpeg'))

x_test_normal = glob(os.path.join(test_normal_dir, '*.jpeg'))
x_test_pneumonia = glob(os.path.join(test_pneumonia_dir, '*.jpeg'))

x_val_normal = glob(os.path.join(val_normal_dir, '*.jpeg'))
x_val_pneumonia = glob(os.path.join(val_normal_dir, '*.jpeg'))

if __name__ == '__main__':
    print(len(x_train_normal)+len(x_train_pneumonia))
