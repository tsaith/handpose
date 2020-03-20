#from skimage.transform import (rescale, resize, rotate,
#                               warp, SimilarityTransform)

from skimage import transform as tf

def rescale(image, scale):
    """
    Scale image by a certain factor.
    """
    return tf.rescale(image, scale)

def resize(image, output_shape):
    """
    Resize image to match a certain size.
    """
    return tf.resize(image, output_shape)

def rotate(image, angle):
    """
    Rotate image by a certain angle around its center.
    """
    return tf.rotate(image, angle)

def shift(image, translation):
    """
    Shift image.
    """
    shift_x = -translation[0]
    shift_y = -translation[1]
    tform = tf.SimilarityTransform(translation=(shift_x, shift_y))
    shifted = tf.warp(image, tform)

    return shifted
