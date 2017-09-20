import numpy as np
from .madgwickahrs import MadgwickAHRS
from .quaternion import Quaternion

class SensorFusion:
    """
    Sensor fusion.
    """

    def __init__(self, dt=0.1, beta=0.041, num_iter=100):
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

        self._fuse = init_madgwick(dt, beta, num_iter=100)
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
            qs.append(self.fuse.quaternion)

        return qs

    def update_ahrs(self, gyro, accel, mag):
        """
        Update of fusion with 9-axis motion sensor.
        """

        qs = []
        for i in range(self.num_iter):
            self.fuse.update_ahrs(gyro, accel, mag)
            qs.append(self.fuse.quaternion)

        return qs


    @property
    def fuse(self):
        return self._fuse

    @fuse.setter
    def fuse(self, fuse_in):
        self._fuse = fuse_in

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
        return self.fuse.quaternion

    @quat.setter
    def quat(self, value):
        self.fuse.quaternion = value

def init_madgwick(dt, beta, num_iter=100):

    sub_dt = dt / num_iter
    q0 = Quaternion([1, 0, 0, 0])
    fuse = MadgwickAHRS(sampleperiod=sub_dt,
                        quaternion=q0, beta=beta)

    return fuse


