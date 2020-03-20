import numpy as np

def to_radian(angle):
    """
    Convert the angle in degree to radian.

    Parameters
    ----------
    angle: array-like
        Angle in degree.

    Returns
    -------
    angle_out: array-like
        Angle in radain.

    """
    angle_out = angle*np.pi/180.0 

    return angle_out

def to_degree(angle):
    """
    Convert the angle in radian to degree.

    Parameters
    ----------
    angle: array-like
        Angle in radain.

    Returns
    -------
    angle_out: array-like
        Angle in degree.
    """
    angle_out = angle * 180.0 / np.pi

    return angle_out
