import pytest
import numpy as np
from numpy.testing import assert_allclose
from handpose.utils import Quaternion


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

def test_roll_pitch_yaw():
    # Test the Tait-Bryan angles estimated from quaternion

    angle_gt = 30.0/180*np.pi
    # Roll
    quat = Quaternion.from_angle_axis(angle_gt, 1, 0, 0)
    roll = quat.roll
    assert_allclose(roll, angle_gt)

    # Pitch
    quat = Quaternion.from_angle_axis(angle_gt, 0, 1, 0)
    pitch = quat.pitch
    assert_allclose(pitch, angle_gt)

    # Yaw
    quat = Quaternion.from_angle_axis(angle_gt, 0, 0, 1)
    yaw = quat.yaw
    assert_allclose(yaw, angle_gt)

def test_quat_vs_array():

    a_gt = np.array([1.0, 1.0, 1.0, 1.0])
    q = Quaternion.from_array(a_gt)
    a = q.to_array()

    assert_allclose(a, a_gt)


def test_quat_vs_spherical():

    theta = 30.0 /180*np.pi
    phi = 60.0 /180*np.pi

    q = Quaternion.from_spherical(theta, phi)
    spherical = q.to_spherical()
    spherical_gt = np.array([theta, phi])

    assert_allclose(spherical, spherical_gt)

if __name__ == '__main__':
    pytest.main([__file__])
