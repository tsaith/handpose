import numpy as np
import matplotlib.pyplot as plt
import skimage
import scipy


def imread(fname):
    """
    Read an image from a file as an array.
    """
    return skimage.io.imread(fname)

def imsave(fname, arr):
    # Save the array as an image file

    return skimage.io.imsave(fname, arr)

def imshow(image):
    # show image

    ndim = image.ndim

    if ndim == 3: # color
        out = plt.imshow(image)
    elif ndim == 2: # gray-scale
        out = plt.imshow(image, cmap = plt.cm.gray)
    else:
        print("Error: ndim = ", ndim)

    return out
