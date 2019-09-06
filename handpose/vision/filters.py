import numpy as np
from scipy.ndimage import filters

def conv2(f, kernel):
    # return g = f conv kernel

    rows, cols = f.shape
    rows_k, cols_k = kernel.shape
    dtype = f.dtype
    g = np.zeros(shape = (rows, cols), dtype = dtype)
    for i in np.arange(1, rows-1):
        for j in np.arange(1, cols-1):
            for ii in np.arange(rows_k):
                for jj in np.arange(cols_k):
                    g[i, j] += f[i+ii-1, j+jj-1]*kernel[ii, jj]

    return g

def convolve(f, kernel):

    g = filters.convolve(f, kernel)

    return g

def smooth_average(f):
    # return smooth image by averaging neighborhoods

    kernel = np.array([[0, 0.2, 0], [0.2, 0.2, 0.2], [0, 0.2, 0]])
    g = convolve(f, kernel)

    return g

def sharpen(image):
    # return sharpen image

    f = image.copy().astype('float64')
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    g = convolve(f, kernel)

    return g

def sobel_h(image):
    # return image filtered by sobel method in horizontal direction

    f = image.copy().astype('float64')
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    g = convolve(f, kernel)

    return g

def sobel_v(image):
    # return image filtered by sobel method in vertical direction

    f = image.copy().astype('float64')
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    g = convolve(f, kernel)

    return g
