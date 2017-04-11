import numpy as np

def has_gesture(X):
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
    thresh = 0.7 # Threshold for gesture detection (g)
    noise_level = 0.2
    bound_cells = 50 # Number of boundary cells

    num_dims = 3
    num_features = len(X)
    size = num_features / num_dims

    ax = X[:size]
    ay = X[size:2*size]
    az = X[2*size:3*size]

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
