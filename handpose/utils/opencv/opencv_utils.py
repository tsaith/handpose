import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image



def plot_vfield(image, flow, fmt='BGR', figsize=None, scale=None, edgecolor='b', linewidth=1.0):
    """
    Plot vector field on the image

    Parameters
    ----------
    image: ndarray
        Image.
    flow: ndarray
        Optical flow.
    fmt: string
        Color format; available candidates: 'BGR', 'RGB', 'Gray'
    figsize: tupple
        Figure size.
    scale: float
       Scale of the vector field.

    Returns
    -------
    fig: object
        Figure object.
    """

    height = image.shape[0]
    width = image.shape[1]
    step = int(width / 15)

    X, Y = np.meshgrid(np.arange(0, width, step), np.arange(0, height, step))
    flow_x = flow[::step, ::step, 0]
    flow_y = flow[::step, ::step, 1]

    fig = plt.figure(figsize=figsize)
    img = imshow(image, fmt)
    ax = fig.axes
    ax[0].quiver(X, Y, flow_x, flow_y, units='width', scale=scale, edgecolor=edgecolor, linewidth=linewidth)

    return fig



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

def imshow(image, fmt='gray'):
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

