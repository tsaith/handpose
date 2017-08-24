import numpy as np

def is_motional(image, image_prev, area_ratio=0.9, pixel_ratio=0.1, max_value=255, verbose=0):
    """
    Detect if the object is motional from two images.
    """

    diff = image - image_prev
    x = np.abs(diff) > max_value*pixel_ratio
    sigma = np.mean(x)

    if sigma > area_ratio:
        status = 1
    else:
        status = 0

    if verbose:
        print("sigma = {}".format(sigma))

    return status

def motion_predict(images, verbose=0):
    """
    Predict motion status from images.
    """

    num_samples = len(images)
    num_detect = num_samples - 1
    status = np.zeros(num_samples-1, dtype=np.int32)

    for i in range(num_detect):
        image = images[i+1]
        image_prev = images[i]
        status[i] = is_motional(image, image_prev)

    num_motional = np.sum(status)
    num_thresh = 0.1*num_detect
    result = 1 if num_motional > num_thresh else 0

    if verbose > 0:
        print("num_motional, threshold = {}, {}".format(num_motional, num_thresh))

    return result
