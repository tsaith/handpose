import numpy as np

def has_gesture(X, thresh = 0.2):
    """
    Detect if it has a gesture.

    Parameters
    ----------
    X: array
        Input features.

    Returns
    -------
    out : bool
        True when it has gesture else False.
    """
    noise_level = 0.2
    bound_cells = 50 # Number of boundary cells

    num_dims = 3
    num_features = len(X)
    size = int(num_features / num_dims)

    ax = X[:, 0]
    ay = X[:, 1]
    az = X[:, 2]

    mag = np.sqrt(ax*ax + ay*ay + az*az)
    mag_static = 1.0 # Static magnitude of acceleration is 1(g)
    diff_abs = np.abs(mag - mag_static)

    out = True if np.max(diff_abs) > thresh else False

    # Mean values of boundaries should be smaller than noise level
    if np.mean(diff_abs[:bound_cells]) > noise_level:
        out = False
    if np.mean(diff_abs[-bound_cells:]) > noise_level:
        out = False

    return out
