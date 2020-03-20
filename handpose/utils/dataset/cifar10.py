import numpy as np
import matplotlib.pyplot as plt

def cifar10_preprocess(data):

    # Normalization
    data = data.astype('float32')
    data /= 255

    return data

def cifar10_combine_images(images):
    # Combine images as a large one.

    combined = (np.concatenate([r.reshape(32, 32, 3) for r in np.split(images, 10)], axis=-1) * 127.5 + 127.5).astype(np.uint8)

    return combined

