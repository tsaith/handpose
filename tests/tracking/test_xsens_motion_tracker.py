import pytest
import numpy as np
from numpy.testing import assert_allclose
from handpose.tracking import *
from handpose.sensor_fusion import *
from handpose.utils import Quaternion


def test_xsens_motion_tracker():

    dt = 1e-2 # Sample period in seconds
    num_iter = 1000
    beta = 0.041 # The suggested beta is 0.041 in Madgwick's paper
    fast_version = True

    # Earth magnetic strength and dip angle
    earth_mag=36.0

    earth_dip_deg=30.0
    earth_dip = earth_dip_deg*np.pi/180


    # Sensor orientation
    angle_deg = 30.0
    angle = angle_deg*np.pi/180

    # Rotaion quaternion, sensor axes respective to user axes
    q_se = Quaternion.from_angle_axis(angle, 0, 1, 0)
    angle_axis_in = q_se.to_angle_axis()

    num_dim = 3

    # Angular rate
    gyro_e = np.zeros(num_dim)
    gyro_e[0] = 0.0
    gyro_e[1] = 0.0 #angle/dt
    gyro_e[2] = 0.0

    # Analytic dynamic acceleration in Earth axes
    accel_dyn_e_in = np.zeros(num_dim)
    accel_dyn_e_in[0] = 0.00 # Dynamic acceleration in earth axes
    accel_dyn_e_in[1] = 0.00
    accel_dyn_e_in[2] = 0.00

    # IMU simulator
    imu_simulator = IMUSimulator(gyro_e, accel_dyn_e_in, q_se, earth_mag=earth_mag, earth_dip=earth_dip)
    gyro_in, accel_in, mag_in = imu_simulator.get_imu_data()

    # Prepare the video
    timesteps = 5
    rows = 48
    cols = 48
    channels = 1
    shape = (timesteps, rows, cols, channels)
    video = np.zeros(shape, dtype=np.int32)
    video = None

    # Eestimate the quaternion
    sf = SensorFusion(dt, beta, num_iter=num_iter, fast_version=fast_version)
    sf.update_ahrs(gyro_in, accel_in, mag_in)
    quat = sf.quat

    # Xsens motion tracker
    tracker = XsensMotionTracker(accel_th=1e-4, vel_th=1e-5)

    q_se_simu = quat

    tracker.update(gyro_in, accel_in, mag_in, accel_dyn=accel_dyn_e_in,
                   motion_status=1, quat=q_se_simu)

    # Test the estimated position
    x_gt = np.array([np.cos(angle), 0, -np.sin(angle)])

    assert_allclose(tracker.x, x_gt, atol=1e-3)

    # Test orientation angles
    roll, pitch, yaw = tracker.get_orientation_angles()
    roll_gt = 0.0
    pitch_gt = angle
    yaw_gt = 0.0

    assert_allclose(roll, roll_gt, atol=1e-3)
    assert_allclose(pitch, pitch_gt, atol=1e-3)
    assert_allclose(yaw, yaw_gt, atol=1e-3)

if __name__ == '__main__':
    pytest.main([__file__])
