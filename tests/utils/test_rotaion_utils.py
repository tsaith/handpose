import pytest
import numpy as np
from numpy.testing import assert_allclose
from handpose.utils import *

def test_quat_vs_array():

    a_gt = np.array([1.0, 1.0, 1.0, 1.0])
    q = quat_from_array(a_gt)
    a = quat_to_array(q)

    assert_allclose(a, a_gt)


def test_quat_vs_spherical():

    theta = 30.0 /180*np.pi
    phi = 60.0 /180*np.pi

    q = quat_from_spherical(theta, phi)
    spherical = quat_to_spherical(q)
    spherical_gt = np.array([theta, phi])

    assert_allclose(spherical, spherical_gt)


if __name__ == '__main__':
    pytest.main([__file__])
