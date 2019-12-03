from PIL import Image
import numpy as np
from numpy import savetxt
import matplotlib.pyplot as plt
from get_data.gather_data import *
import cv2
from skimage.feature import greycomatrix, greycoprops
import pywt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis


def get_image(path, verbose=False):
    im = np.array(Image.open(path).resize((256, 256), Image.LANCZOS).convert("L"))
    if verbose:
        print(im.shape)
        print(np.amax(im))
        print(im.dtype)
        plt.imshow(im, cmap= plt.get_cmap('gray'), origin="lower")
        plt.show()
    return im


def equal_hist(img, filter="median"):
    if filter == "median":
        img = cv2.medianBlur(img, ksize=3)
    # Method found on opencv.org
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    return cl1


def features_texture_extraction(img):
    glcm = greycomatrix(img, [1], [0, np.deg2rad(45), np.deg2rad(90), np.deg2rad(135)])
    contrast = np.squeeze(greycoprops(glcm, 'contrast'))
    correlation = np.squeeze(greycoprops(glcm, 'correlation'))
    energy = np.squeeze(greycoprops(glcm, 'energy'))
    homogeneity = np.squeeze(greycoprops(glcm, 'homogeneity'))
    features = np.concatenate((contrast, correlation, energy, homogeneity))
    return features


def features_frequency_extraction(img):
    """

    :param img:
    :return:
    """
    shape = img.shape
    max_lev = 4  # how many levels of decomposition
    c = pywt.wavedec2(img, 'db2', mode='periodization', level=max_lev)
    feature_vect = [np.mean(c[0]), np.std(c[0])]
    # As wavedec2 returns coefficient in descending order, we don't look at last element:
    for i in range(1, len(c)-1):
        for elt in c[i]:
            feature_vect.append(np.mean(elt))
            feature_vect.append(np.std(elt))
    return np.array(feature_vect)


def dw_show(img):
    shape = img.shape
    max_lev = 3  # how many levels of decomposition to draw
    label_levels = 3  # how many levels to explicitly label on the plots
    fig, axes = plt.subplots(2, 4, figsize=[14, 8])
    for level in range(0, max_lev + 1):
        if level == 0:
            # show the original image before decomposition
            axes[0, 0].set_axis_off()
            axes[1, 0].imshow(img, cmap=plt.cm.gray)
            axes[1, 0].set_title('Image')
            axes[1, 0].set_axis_off()
            continue
        # plot subband boundaries of a standard DWT basis
        draw_2d_wp_basis(shape, wavedec2_keys(level), ax=axes[0, level], label_levels=label_levels)
        axes[0, level].set_title('{} level\ndecomposition'.format(level))
        # compute the 2D DWT
        c = pywt.wavedec2(img, 'db2', mode='periodization', level=level)
        print(level, np.array(c[0]).shape)
        # normalize each coefficient array independently for better visibility
        c[0] /= np.abs(c[0]).max()
        for detail_level in range(level):
            c[detail_level + 1] = [d / np.abs(d).max() for d in c[detail_level + 1]]
        # show the normalized coefficients
        arr, slices = pywt.coeffs_to_array(c)
        axes[1, level].imshow(arr, cmap=plt.cm.gray)
        axes[1, level].set_title('Coefficients\n({} level)'.format(level))
        axes[1, level].set_axis_off()
    plt.tight_layout()
    plt.show()


def get_features(img, verbose=False):
    if verbose:
        # Faire un subplot
        plt.imshow(equal_hist(img))
        plt.title("Avec egalisation de l'histo")
        plt.figure()
        plt.imshow(img)
        plt.title("Raw image")
        plt.show()
    img = equal_hist(img)
    texture_features = features_texture_extraction(img)
    freq_features = features_frequency_extraction(img)
    features_vector = np.concatenate((texture_features, freq_features))
    return features_vector


def save_img_dataset_to_feature(path, name):
    print(f"{name} en cours de traitement")
    print(f"{len(path)} images to convert and save...")
    data_set_features = []
    for elt in path:
        current = get_image(elt)
        data_set_features.append(get_features(current))
    path_to_save = os.path.join("..", "Data", "chest_xray_features_vect", name)
    np.save(path_to_save, np.array(data_set_features))


if __name__ == '__main__':

    def convert_all_datas():
        save_img_dataset_to_feature(x_train_normal, "x_train_normal")
        save_img_dataset_to_feature(x_test_normal, "x_test_normal")
        save_img_dataset_to_feature(x_val_normal, "x_val_normal")
        save_img_dataset_to_feature(x_train_pneumonia, "x_train_pneumonia")
        save_img_dataset_to_feature(x_test_pneumonia, "x_test_pneumonia")
        save_img_dataset_to_feature(x_val_pneumonia, "x_val_pneumonia")

    convert_all_datas()


