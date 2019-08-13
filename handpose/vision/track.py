from .helpers import sliding_windows


def detect_windows(image, target):
    """
    Windows for object detection.

    Parameter
    ---------

    image : array-like, shape (rows, cols).
        Input image with gray scale.

    target : defined object.
        Rectangular target.

    Return
    ------

    wins : list
        Windows for object detection.

    """

    target_rows, target_cols = target.shape

    scales = [1.2, 1.0, 0.8]
    step = 100

    wins = []
    for scale in scales:
        scaled_rows = int(scale*target_rows)
        scaled_cols = int(scale*target_cols)
        scaled_shape = (scaled_rows, scaled_cols)
        scaled_wins = sliding_windows(image, win_shape=scaled_shape, step=step)
        wins += scaled_wins

    return wins
