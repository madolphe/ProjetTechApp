#@TODO pre-processing pour extraction de characteristiques (à développer)

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from get_data.gather_data import *


def get_image(path,verbose =False):
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


if __name__ == '__main__':
    current = get_image(x_train_pneumonia[100], verbose=False)
    mean = mean_image()
    histogram2d(mean, current)





