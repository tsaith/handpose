import numpy as np
import os
import cv2

from . import transform as tf
from .core import SlidingWindow
from .io import imsave

def sliding_windows(image, win_shape, step):
    """
    Sliding windows.

    Parameters
    ----------

    image: array-like, shape (rows, cols)
        Input image

    win_shape: tuple
        window shape.

    step: integer
        Shift step of window.

    Return
    ------

    wins: list
        List of sliding windows.
    """

    image_h, image_w = image.shape
    win_h, win_w = win_shape

    wins = []
    for iy in range(0, image_h-win_h, step):
        for ix in range(0, image_w-win_w, step):
            win_image = image[iy:iy + win_h, ix:ix + win_w]
            win = SlidingWindow(ix, iy, win_w, win_h, win_image)
            wins.append(win)

    return wins

def pyramid(image, scale=0.8, shape_mini=(20, 20)):
    """
    Return image pyramid which consists of the images scaled.
    """

    w_min = shape_mini[0] # Width
    h_min = shape_mini[1] # Height

    scaled = image
    yield scaled

    while True:

        scaled = rescale(scaled, scale)
        w, h = scaled.shape

        if  w < w_min or h < h_min:
            break

        yield scaled

def save_sliding_windows(wins, image, dir_path='.'):
    """
    Draw box on each image iteratively and save them to the directpry specified.
    """

    n_wins = len(wins)

    for i in range(n_wins):
        img = image.copy()
        fname = "image_" + str(i) + ".png"
        fpath = os.path.join(dir_path, fname)
        win = wins[i]
        cv2.rectangle(img, (win.x0, win.y0), (win.x0 + win.width, win.y0 + win.height), (0, 255, 0), thickness=2)
        print("Save image ", fpath)
        imsave(fpath, img)
