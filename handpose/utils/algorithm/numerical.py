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

