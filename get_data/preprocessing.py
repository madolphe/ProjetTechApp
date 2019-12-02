# @TODO pre-processing pour extraction de characteristiques (à développer)

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from get_data.gather_data import *
import cv2
from skimage.feature import greycomatrix, greycoprops
import pywt

def get_image(path, verbose=False):
    im = np.array(Image.open(path).resize((256, 256), Image.LANCZOS).convert("L"))
    if verbose:
        print(im.shape)
        print(np.amax(im))
        print(im.dtype)
        plt.imshow(im, cmap= plt.get_cmap('gray'), origin="lower")
        plt.show()
    return im


def mean_image():
    mean_img = np.mean(np.array([get_image(element) for element in x_train_pneumonia]), axis=0)
    # plt.imshow(mean_img)
    # plt.show()
    return mean_img


def histogram2d(mean_img, current_img):
    hist, _, _ = np.histogram2d(mean_img.flatten(), current_img.flatten(), bins=100)
    print(type(hist))
    plt.imshow(hist)
    plt.show()


def sifting(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img,None)
    img = cv2.drawKeypoints(img,kp,None)
    cv2.imwrite('sift_keypoints.jpg', img)
    cv2.imshow("h",img)


def fourier(img):

    transform_image = np.fft.rfft2(img)
    shift = np.fft.fftshift(transform_image)
    magnitude_spectrum = 20*np.log(np.abs(shift))
    magnitude_spectrum = np.resize(magnitude_spectrum, (256, 256))

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    return magnitude_spectrum


def equal_hist(img, filter="median"):
    if filter == "median":
        img = cv2.medianBlur(img, ksize=3)
    # Methode trouvée sur openCV
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


def features_complexity_extraction():
    pass


from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis


def features_frequency_extraction(img):
    # res = pywt.wavedec2(img, 'db1', 4)
    # print(type(res))
    # print(res[3])
    # titles = ['Approximation', ' Horizontal detail',
    #           'Vertical detail', 'Diagonal detail']
    # coeffs2 = pywt.dwt2(img, 'bior1.3')
    # LL, (LH, HL, HH) = coeffs2
    # print(LL.shape)
    # fig = plt.figure(figsize=(12, 3))
    # for i, a in enumerate([LL, LH, HL, HH]):
    #     ax = fig.add_subplot(1, 4, i + 1)
    #     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    #     ax.set_title(titles[i], fontsize=10)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    # fig.tight_layout()
    # plt.show()
    # On prendra moyenne et variance et ce sera all good

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
    return


if __name__ == '__main__':
    current = get_image(x_train_normal[100], verbose=False)
    # texture_features = features_texture_extraction(current)
    # features_frequency_extraction(current)
    plt.imshow(equal_hist(current))
    plt.title("Avec egalisation de l'histo")
    plt.figure()
    plt.imshow(current)
    plt.title("Raw image")
    plt.show()
    # sifting(current)
    # fou = fourier(current)
    # mean = mean_image()
    # histogram2d(mean, current)






