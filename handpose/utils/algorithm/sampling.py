import numpy as np

def downsample(data, step=1):
    """
    Downsampling.

    Parameters
    ----------
    data: array-like
        Input data.
    step: int
        Step used to pick signals.
    """

    num_samples = len(data)

    out = [data[i, ::step] for i in range(num_samples)]
    out = np.asarray(out)

    return out

def sampling_window_frames(data, ia, size, hop):
    """
    Sampling window frames.

    ----------
    Parameters

    data: ndarray
        Input data.
    ia: int
        Index of the first window frame.
    size: int
        Size of window frame.
    hop: int
        Hop between window frames.

    -------
    Returns

    wins: array in 2D
        Window frames in shape of (num, size)
    """

    # Number of data
    num_data = len(data)

    # Initial indexes of window frames
    iz = num_data - size + 1
    indexes = range(ia, iz, hop)

    # Number of window frames
    num = len(indexes)

    # Allocate array of window frames
    wins = np.zeros((num, size))

    # Prepare data for window frames
    for i in indexes:
        wins[i] = data[i:i+size].copy()

    return wins
