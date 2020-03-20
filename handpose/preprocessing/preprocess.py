import numpy as np

def preprocess(data, calc_mag=True, shift_baseline=True, normalize=True):
    """
    Preprocess the data.
    """

    arr_min = np.min(data, axis=1)
    arr_max = np.max(data, axis=1)
    arr_range = arr_max - arr_min
    out = data.copy()

    # Get the magnitude
    if calc_mag:
        out = np.sqrt(data[:, 0::2]**2 + data[:, 1::2]**2)

    # Shift the distribution in magnitude
    if shift_baseline:
        out = out - arr_min[:, np.newaxis]

    # Nomalization
    if normalize:
        out = out / arr_range[:, np.newaxis]

    return out


