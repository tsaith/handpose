import numpy as np
from .madgwick import FastMadgwick, MadgwickAHRS

class SensorFusion:
    """
    Sensor fusion.
    """

    def __init__(self, dt=0.1, beta=0.041, num_iter=1):
        """
        Constructor of SensorFusion.

        Parameters
        ----------
        dt: float
            Sampling time duration in seconds.
        beta: float
            Fusion parameter.
        num_iter: int
            Number of iterations.
        """

        sub_dt = dt / num_iter
        #self._fuse = FastMadgwick(dt, beta)
        self._fuse = MadgwickAHRS(sub_dt, beta)

        self._dt = dt
        self._beta = beta
        self._num_iter = num_iter

    def update_imu(self, gyro, accel):
        """
        Update of fusion with 6-axis IMU.
        """
        qs = []
        for i in range(self.num_iter):
            self.fuse.update_imu(gyro, accel)
            qs.append(self.fuse.quat)

        return qs

    def update_ahrs(self, gyro, accel, mag):
        """
        Update of fusion with 9-axis motion sensor.
        """
        qs = []
        for i in range(self.num_iter):
            self.fuse.update_ahrs(gyro, accel, mag)
            qs.append(self.fuse.quat)

        return qs

    @property
    def fuse(self):
        return self._fuse

    @fuse.setter
    def fuse(self, value):
        self._fuse = value

    @property
    def dt(self):
        return self._dt

    @property
    def beta(self):
        return self._beta

    @property
    def num_iter(self):
        return self._num_iter

    @property
    def quat(self):
        return self.fuse.quat


