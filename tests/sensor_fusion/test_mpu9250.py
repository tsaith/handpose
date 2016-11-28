import pytest
import numpy as np
from numpy.testing import assert_allclose
from handpose.sensor_fusion import MPU9250


def test_convert_axes():

    num_dim = 3 # Number of dimensions

    # Raw readings
    gyro_raw  = np.zeros(num_dim)
    accel_raw = np.zeros(num_dim)
    mag_raw   = np.zeros(num_dim)

    gyro_raw[0] = 1
    gyro_raw[1] = 2 
    gyro_raw[2] = 3 

    accel_raw[0] = 1
    accel_raw[1] = 2 
    accel_raw[2] = 3 

    mag_raw[0] = 1
    mag_raw[1] = 2 
    mag_raw[2] = 3 

    # Readings in the sensor axes
    gyro_s  = np.zeros(num_dim)
    accel_s = np.zeros(num_dim)
    mag_s   = np.zeros(num_dim)

    gyro_s[0] =  gyro_raw[0]
    gyro_s[1] = -gyro_raw[1]
    gyro_s[2] = -gyro_raw[2]

    accel_s[0] =  accel_raw[0]
    accel_s[1] = -accel_raw[1]
    accel_s[2] = -accel_raw[2]

    mag_s[0] =  mag_raw[1]
    mag_s[1] = -mag_raw[0]
    mag_s[2] =  mag_raw[2]

    imu_s = [gyro_s, accel_s, mag_s]

    # Initialize IMU object
    mpu9250 = MPU9250()
    # Convert to the sensor axes
    imu = mpu9250.convert_axes(gyro_raw, accel_raw, mag_raw)

    assert_allclose(imu, imu_s)


if __name__ == '__main__':
    pytest.main([__file__])
