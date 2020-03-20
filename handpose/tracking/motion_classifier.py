import numpy as np

def is_motional(image, image_prev, area_ratio=0.2, pixel_ratio=0.01, max_value=255, verbose=0):
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

def predict_unit(images, verbose=0):
    """
    Prediction for unit instance.
    """

    timesteps, rows, cols, channels = images.shape

    num_detect = timesteps - 1
    status = np.zeros(num_detect, dtype=np.int32)

    for i in range(num_detect):
        image = images[i+1]
        image_prev = images[i]
        status[i] = is_motional(image, image_prev)

    num_motional = np.sum(status)
    num_thresh = 0.2*num_detect
    result = 1 if num_motional > num_thresh else 0

    if verbose > 0:
        print("num_motional, threshold = {}, {}".format(num_motional, num_thresh))

    return result

def motion_predict(data, verbose=0):
    """
    Motion prediction.
    """
    num_samples, time_steps, rows, cols, channels = data.shape
    labels = np.zeros(num_samples, dtype=np.int32)

    for i in range(num_samples):
        images = data[i]
        labels[i] = predict_unit(images)

    return labels
