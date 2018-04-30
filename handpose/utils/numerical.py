import numpy as np

def absolute_errors(predictions, targets):
    """
    Absolute errors.
    """
    return predictions - targets

def relative_errors(predictions, targets):
    """
    Relative errors.
    """

    err_tol = 1e-10 # Error tolerance
    rel_errs = (predictions - targets) / (targets + err_tol)

    return rel_errs

def maximum_multi(arrays):
    """
    Find the element-wise maximum values from multiple arrays.
    """

    num_arrays = len(arrays)
    out = arrays[0]

    for i in range(1, num_arrays):
        out = np.maximum(out, arrays[i])

    return out

def fft_windows(wins):
    """
    Perform FFT on each window frame.
    """

    num_wins, size = wins.shape

    # Allocate spectrum array
    spectra = np.zeros((num_wins, size), dtype=complex)

    # Perform FFTs
    for i in range(num_wins):
        y = wins[i, :]
        Y = np.fft.fft(y) ## FFT
        Y /= size

        spectra[i] = Y

    return spectra


#def interp1d(data, num_out):
#    """
#    Interpolation of 1D data.
#
#    Parameters
#    ----------
#    data: array
#        Data array of 1D.
#    num_out: int
#        number of data after interpolation.
#
#    Returns
#    -------
#    y_out: array
#        Data array after interpolation.
#    """
#
#    num_in = len(data)
#
#    xa = 0.0
#    xz = 1.0
#    x_in = np.linspace(xa, xz, num=num_in)
#    y_in = data
#
#    x_out = np.linspace(xa, xz, num=num_out)
#    y_out = np.interp(x_out, x_in, y_in)
#
#    return y_out
#
#def interp3d(data, num_out):
#    """
#    Interpolation of 3D data.
#
#    Parameters
#    ----------
#    data: array
#        Data array of 3D.
#    num_out: int
#        number of data after interpolation.
#
#    Returns
#    -------
#    y_out: array
#        Data array after interpolation.
#    """
#
#    out = np.zeros((num_out, 3))
#    out[:, 0] = interp1d(data[:, 0], num_out)
#    out[:, 1] = interp1d(data[:, 1], num_out)
#    out[:, 2] = interp1d(data[:, 2], num_out)
#
#    return out
