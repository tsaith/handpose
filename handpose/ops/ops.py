
import tensorflow as tf
import numpy as np

def concat(tensors, axis=0):
    # Concatenat tensors.
    return tf.concat(tensors, axis)

def to_categorical_images(y, height, width):
    """
    Convert the categorical labels into images.
    """
    num = y.shape[0]
    channels = y.shape[1]
    print(channels)

    images = tf.reshape(y, (-1, 1, 1, channels))
    images = images*tf.ones([num, height, width, channels])

    return images

