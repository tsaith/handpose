import numpy as np
import os
from ..io import imread

def load_train_images(dir_path):
    """
    Load training images.

    Return
    ------
    X : array-like, shape (n_samples, n_features)
        Training data.

    y : array-like, shape (n_samples)
        Training class labels.
    """

    files = os.listdir(dir_path)

    images = []
    for f in files:
        if f.endswith('.png') or f.endswith('.gif'):
            images.append(f)

    n_samples = len(images)
    arr = imread(dir_path + '/' + images[0])
    dt = arr.dtype
    rows, cols = arr.shape
    n_features = rows * cols

    X = np.zeros((n_samples, n_features), dtype=dt)
    y = np.zeros((n_samples), dtype=np.str)
    for i in np.arange(n_samples):
        image = images[i]
        image_path = dir_path + '/' + image
        arr = imread(image_path)
        X[i, :] = imread(image_path).flatten()
        y[i] = image.split('_')[1]

    return X, y
