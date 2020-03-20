import pytest
import numpy as np
from numpy.testing import assert_allclose
from handpose.utils import *


def test_earth_to_screen_axes():

    vec0 = np.array([1.0, 2.0, 3.0])
    vec = earth_to_screen_axes(vec0)
    vec_gt = np.array([2.0, 3.0, 1.0])

    assert_allclose(vec, vec_gt)

if __name__ == '__main__':
    pytest.main([__file__])
