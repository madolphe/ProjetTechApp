from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from get_data.gather_data import *
import cv2
from skimage.feature import greycomatrix, greycoprops
import pywt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis


def get_image(path, verbose=False):
    """
    Method to get a single image
        - Crop to a 256x256 image with Lanczos interpollation
    :param path:
    :param verbose: tell if you want more infos on your image
    :return:
    """
    im = np.array(Image.open(path).resize((256, 256), Image.LANCZOS).convert("L"))
    if verbose:
        print(im.shape)
        print(np.amax(im))
        print(im.dtype)
        plt.imshow(im, cmap=plt.get_cmap('gray'), origin="lower")
        plt.show()
    return im


def equal_hist(img, filter="median"):
    """
    Enhance img quality (improve contrast)
    :param img:
    :param filter:
    :return:
    """
    if filter == "median":
        img = cv2.medianBlur(img, ksize=3)
    # Method found on opencv.org
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    return cl1


def features_texture_extraction(img):
    """
    Compute grey-level co-occurence matrix for distance = 1 and 4 diffrent direction = 0, 45, 90, 135.
    Then compute all metrics based on glcm (contrast, correlation, energy and homogeneity).
    :param img:
    :return:
    """
    glcm = greycomatrix(img, [1], [0, np.deg2rad(45), np.deg2rad(90), np.deg2rad(135)])
    contrast = np.squeeze(greycoprops(glcm, 'contrast'))
    correlation = np.squeeze(greycoprops(glcm, 'correlation'))
    energy = np.squeeze(greycoprops(glcm, 'energy'))
    homogeneity = np.squeeze(greycoprops(glcm, 'homogeneity'))
    features = np.concatenate((contrast, correlation, energy, homogeneity))
    # feature vect =
    # ['contrast_0', 'contrast_45', 'contrast_90', 'contrast_135',
    # 'correlation_0', 'correlation_45', 'correlation_90', 'correlation_135',
    # 'energy_0', 'energy_45', 'energy_90', 'energy_135',
    # 'homogeneity_0', 'homogeneity_45', 'homogeneity_90', 'homogeneity_135']
    return features


def features_frequency_extraction(img):
    """
    Compute wavelet decomposition and mean/std for each level and each direction of wavelet
    Note: one level is composed of 4 images --> a low frequence one, 3 high frequence with 3 directions
    horizontal, vertical, diagonal.
    Feature vect is only composed of mean/std of last level for low frequency and third last level for high freq.
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
    # feature vect =
    # [mean4LF, std4LF, mean3HF_H, std3HF_H, mean3HF_V, std3HF_V,
    #  mean3HF_D, std3HF_D, mean2HF_H, std2HF_H, mean2HF_V, std2HF_V,
    #  mean2HF_D, std2HF_D, mean1HF_H, std1HF_H, mean1HF_V, std1HF_V,
    #  mean1HF_D, std1HF_D]
    return np.array(feature_vect)


def dw_show(img):
    """
    Code taken from https://pywavelets.readthedocs.io/ to show what wavelet decomposition does.
    :param img:
    :return:
    """
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
    """
    Method used to create final feature vect for an input image
    :param img:
    :param verbose:
    :return:
    """
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
    """
    Method used to save in npy file a set of images encoded with our feature extraction method
    :param path:
    :param name:
    :return:
    """
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

    # convert_all_datas()
    dw_show(get_image(x_train_normal[0]))

