import pytest
import numpy as np
from numpy.testing import assert_allclose
from handpose.tracking import *

def test_motion_predict():

    num_samples = 2
    timesteps = 5
    height, width = 2, 2
    channels = 1 # Gray-scale

    data = np.zeros((num_samples, timesteps, height, width, channels))

    # Smaple 1
    data[0, 0, :, :, :] = 0
    data[0, 2, :, :, :] = 0
    data[0, 4, :, :, :] = 0

    # Sample 2
    data[1, 0, :, :, :] = 255
    data[1, 2, :, :, :] = 0
    data[1, 4, :, :, :] = 255

    labels = motion_predict(data)
    labels_gt = [0, 1]

    assert_allclose(labels, labels_gt)


if __name__ == '__main__':
    pytest.main([__file__])
