import numpy as np


def earth_to_screen_axes(vec):
    """
    Return the vector of the Earth axes to that of the screen axes.

    Parameters
    ----------
    vec: array-like
        Vector in Earth axes.

    Returns
    -------
    out: array-like
        Vector in the screen axes.
    """
    out = np.zeros_like(vec)

    # Rotate the vector with
    # x' = y, y' = z, z' = x
    out[0] = vec[1]
    out[1] = vec[2]
    out[2] = vec[0]

    return out
