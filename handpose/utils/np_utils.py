import numpy as np

def est_mag(vec):
    """
    Estimate the magnitude of a vector.
    """
    return np.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])
