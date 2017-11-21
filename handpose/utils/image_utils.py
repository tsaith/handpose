import numpy as np
import matplotlib.pyplot as plt
import cv2

def color2gray(color, color_seq="BGR", has_channel=True):
    """
    Convert color image as gray one.
    """

    if color_seq == "BGR":
        convert_type = cv2.COLOR_BGR2GRAY
    else:
        convert_type = cv2.COLOR_RGB2GRAY

    gray = cv2.cvtColor(color, convert_type)
    if has_channel:
        gray = np.expand_dims(gray, axis=2)

    return gray

def imshow(image, fmt):
    """
    Show image with gray-scale.

    fmt: string
        Color format; Available candidates: 'BGR', 'RGB', 'Gray'
    """

    if image.ndim == 2: # Gray
        height, width = image.shape
        channels = 1
    else: # Color or gray
        height, width, channels = image.shape

    if fmt == 'BGR' or fmt == 'bgr':
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = plt.imshow(rgb, interpolation='nearest')

    if fmt == 'RGB' or fmt == 'rgb':
        img = plt.imshow(image, interpolation='nearest')

    if fmt == 'Gray' or fmt == 'gray':
        if image.ndim == 2:
            gray = image
        else:
            gray = image[:, :, 0]
        img = plt.imshow(gray, interpolation='nearest', cmap=plt.cm.gray)

    return img

