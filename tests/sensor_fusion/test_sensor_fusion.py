import pytest
import numpy as np
from numpy.testing import assert_allclose
from handpose.sensor_fusion import *


def test_sensor_fusion():


    # Analytic IMU inputs
    gyro = np.array([0.0, 0.0, 0.0])
    accel = np.array([-0.05233596, 0.0, 0.99862953])
    mag = np.array([32.07623487, 0.0, -16.34365799])

    # Estimate the quaternion
    dt = 0.01
    beta = 0.041
    num_iter = 100
    fast_version = True

    fusion = SensorFusion(dt, beta, num_iter, fast_version=fast_version)
    fusion.update_ahrs(gyro, accel, mag)

    q_es_simu = fusion.quat.to_array()
    q_es_gt = np.array([0.99965732, -0.0, -0.02617695, -0.0]) # Ground True

    assert_allclose(q_es_simu, q_es_gt, rtol=1e-2)


if __name__ == '__main__':
    pytest.main([__file__])
