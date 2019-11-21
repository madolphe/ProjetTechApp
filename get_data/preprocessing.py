#@TODO pre-processing pour extraction de characteristiques (à développer)

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from get_data.gather_data import *
import cv2



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


if __name__ == '__main__':
    current = get_image(x_train_normal[100], verbose=False)
    sifting(current)
    # fou = fourier(current)
    # mean = mean_image()
    # histogram2d(mean, current)






