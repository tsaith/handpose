import numpy as np
import matplotlib.pyplot as plt

def plot_vfield(image, flow, figsize=None, scale=None, edgecolor='b', linewidth=1.0):
    """
    Plot vector field on the image

    Parameters
    ----------
    image: ndarray
        Image.
    flow: ndarray
        Optical flow.
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
    ax = fig.axes
    ax[0].quiver(X, Y, flow_x, flow_y, units='width', scale=scale, edgecolor=edgecolor, linewidth=linewidth)

    return fig
