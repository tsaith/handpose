import numpy as np
from scipy import interpolate

def interp1d(x_in, y_in, x_out, kind='linear'):
    """
    Interpolate a 1D function.
    """

    f = interpolate.interp1d(x, y)
    y_out = f(x_out)

    return y_out

