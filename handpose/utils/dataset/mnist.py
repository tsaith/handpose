import numpy as np
import matplotlib.pyplot as plt

def mnist_preprocess(data):

    # Add dimension of channels
    data = np.expand_dims(data, axis=data.ndim+1)

    # Normalization
    data = data.astype('float32')
    data /= 255

    return data

def mnist_combine_images(images):
    # Combine images as a large one.

    combined = (np.concatenate([r.reshape(-1, 28) for r in np.split(images, 10)], axis=-1) * 127.5 + 127.5).astype(np.uint8)

    return combined

