import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_pixel(r, g, b):
    pixel = np.zeros(shape = (3), dtype = np.uint8)
    pixel[0] = r
    pixel[1] = g
    pixel[2] = b

    return pixel

def get_black_pixel():
    pixel = np.zeros(shape = (3), dtype = np.uint8)
    pixel[0] = 255
    pixel[1] = 255
    pixel[2] = 255
    return pixel

def get_white_pixel():
    pixel = np.zeros(shape = (3), dtype = np.uint8)
    pixel[0] = 0
    pixel[1] = 0
    pixel[2] = 0
    return pixel

def rgb2gray(image):
    # Convert rgb to gray
    out = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return out

def gray2rgb(image):
    # Convert rgb to gray
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return out

def gray2binary(image, threshold = 127):
    # convert gray to binary

    rows, cols = image.shape
    out = np.zeros(shape = (rows, cols), dtype=np.bool)
    for row in range(rows):
        for col in range(cols):
            val = image[row,col]
            if val > threshold:
                out[row,col] = True
            else:
                out[row,col] = False

    return out

def rgb2binary(image, threshold = 127):
    # convert rgb to binary

    image_g = rgb2gray(image)
    out = gray2binary(image_g, threshold)

    return out


def binary2gray(image, threshold = 127):
    # convert binary to gray image

    rows, cols = image.shape
    out = np.zeros(shape = (rows, cols), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            val = image[row, col]
            if val == 1:
                out[row, col] = 255
            else:
                out[row, col] = 0

    return out

def bgr2rgb(image):
    """
    Convert color channels from BRG to RGB.
    """
    out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return out

def im2dto3d(image):
    # convert image from 2D to 3D

    dtype = image.dtype
    rows, cols = image.shape
    channels = 3
    out = np.zeros(shape = (rows, cols, channels), dtype = dtype)
    for channel in range(channels):
        out[:,:,channel] = image

    return out

def hist_prob(image, nbins = 256):
    # Return the probability function used in histogram equalization

    rows, cols = image.shape
    total_pixels = rows * cols
    pf = np.zeros(shape = (nbins), dtype = np.float64)
    for row in range(rows):
        for col in range(cols):
            i = image[row, col]
            pf[i] += 1
    pf /= total_pixels

    return pf


def cumulative_distribution(image, nbins = 256):
    # Return cumulative distribution function (cdf)
    # for the given image of grayscale
    # cdf(i,j) = sum_k^I(i,j) pf(k)

    rows, cols = image.shape
    total_pixels = rows * cols
    pf = hist_prob(image, nbins)
    cdf = np.zeros(shape = (rows, cols), dtype = np.float64)
    for row in range(rows):
        for col in range(cols):
            intensity = image[row, col]
            for k in range(intensity):
                cdf[row, col] += pf[k]

    return cdf

def hist_equalization(image, nbins = 256):
    # Return image after histogram equalization
    # for the given image of grayscale

    cdf =  cumulative_distribution(image, nbins)
    out = np.floor((nbins - 1) * cdf)

    return out

def histogram(image, nbins = 256):
    # Return histogram of image

    hist = hist_prob(image, nbins)
    bins = np.arange(nbins, dtype = np.int32)

    return hist, bins
