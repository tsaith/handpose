import numpy as np
import quaternion

def quat_from_array(a):
    # Return quaternion from array.
    return quaternion.as_quat_array(a)

def quat_to_array(q):
    """
    Return the array from quaternion.

    Parameters
    ----------
    q: object
        Quaternion.

    Return
    ------
    a: array
        Array from quaternion.
    """
    return quaternion.as_float_array(q)

def quat_from_angle_axis(rad, x, y, z):
    """
    Return quaternion from angle-axis representation.

    Parameters
    ----------
    rad: float
        Angle in radian.
    x: float
        x coordinate.
    y: float
        y coordinate.
    z: float
        z coordinate.

    Returns
    -------
    quat: object
        Quaternion.
    """
    s = np.sin(0.5*rad)
    return quat_from_array([np.cos(0.5*rad), x*s, y*s, z*s])

def quat_from_spherical(theta, phi):
    """
    Return the quaternion from the spherical coordinates.

    Parameters
    ----------
    theta: float
        Poloidal angle.
    phi: float
        Toroidal angle.

    Returns
    -------
    quat: quaternion
         Rotational quaternion.
    """
    return quaternion.from_spherical_coords(theta, phi)

def quat_to_spherical(q):
    """
    Convert the quaternion into the spherical coordinates.

    Parameters
    ----------
    q: object
       Quaternion.

    Returns
    -------
    coords: array
        Array of spherical coordinates; [theta, phi], where
        theta is the poloidal angle and phi is the toroidal angle.
        The unit z vector is the reference vector.
    """
    return quaternion.as_spherical_coords(q)

