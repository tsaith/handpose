import pytest
import numpy as np
from numpy.testing import assert_allclose
from handpose.sensor_fusion import *


def test_quat_inv():

    # Define the quaternion from angle-axis representation
    angle = 0.5*np.pi # Angle for rotation
    x = 0.0
    y = 1.0
    z = 0.0
    quat = Quaternion.from_angle_axis(angle, x, y, z)
    quat_inv_gt = Quaternion.from_angle_axis(angle, -x, -y, -z) # Ground true
    quat_inv = quat.inv()

    inv_arr_gt = quat_inv_gt.to_array()
    inv_arr = quat_inv.to_array()

    assert_allclose(inv_arr, inv_arr_gt)

def test_rotate_axes():

    # Define the quaternion from angle-axis representation
    angle = 0.5*np.pi # Angle for rotation
    quat = Quaternion.from_angle_axis(angle, 1, 0, 0)

    # Define a vector for testing
    vec_ori = np.array([1.0, 0.0, 0.0])

    # Rotate the vector with quaternion
    vec_rot = quat.rotate_axes(vec_ori)

    assert_allclose(vec_rot, vec_ori)

if __name__ == '__main__':
    pytest.main([__file__])