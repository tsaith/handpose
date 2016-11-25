import numpy as np
from ..sensor_fusion import *


class MotionTracker:
    """
    Motion tracker based on sensor fustion.
    """

    def __init__(self, dt=0.1, beta=0.041, num_iter=100):
        """
        dt: float
            Sampling time duration in seconds.
        beta: float
            Parameter used in Madgwick's algorithm.
        num_iter: int
            Number of iterations used to update sensor fusion.
        """

        self._dt = dt
        self._beta = beta
        self._num_iter = num_iter
        
        num_dim = 3 # Number of dimensions

        # Initialize Madgwick's model
        self._fuse = init_madgwick(self._dt, self._beta, 
                                   num_iter=self._num_iter)

        # Translational information
        self._accel_dyn = np.zeros(num_dim) # Dynamic acceleration
        self._vel = np.zeros(num_dim)
        self._dv = np.zeros(num_dim)
        self._dx = np.zeros(num_dim)

        # Angular information
        self._theta = np.zeros(num_dim)
        self._dtheta = np.zeros(num_dim)

        # IMU information
        self._gyro  = None
        self._accel = None
        self._mag   = None
        
    def update(self, gyro, accel, mag):
        """
        Update the status of tracker.

        Parameters
        ----------
        gyro: array
            Angular rates (rad/s)
        accel: array
            Measured acceleration (m/s^2)
        mag: array
            Measured magnetic fields (muT)
        """
        
        # Update readings from IMU
        self.gyro = gyro
        self.accel = accel
        self.mag = mag

        # Sensor fusion
        imu = [gyro, accel, mag]
        fusion_update(self.fuse, imu, num_iter=self._num_iter)

        # Estimate the dynamic acceleration
        self.accel_dyn = dynamic_accel(accel, self.q) 

        # Update velocity and displacement
        self.dv = self.accel_dyn * self.dt    
        self.vel +=  self.dv
        self.dx = self.vel * self.dt

        # Update angular velocity and displacement
        self.dtheta = self.w * self.dt
        self.theta +=  self.dtheta

    @property
    def fuse(self):
        return self._fuse

    @property
    def q(self):
        return self.fuse.quaternion

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
    def gyro(self):
        return self._gyro
    
    @gyro.setter
    def gyro(self, gyro_in):
        self._gyro = gyro_in

    @property
    def accel(self):
        return self._accel

    @accel.setter
    def accel(self, accel_in):
        self._accel = accel_in

    @property
    def mag(self):
        return self._mag
    
    @mag.setter
    def mag(self, mag_in):
        self._mag = mag_in

    @property
    def accel_dyn(self):
        return self._accel_dyn

    @accel_dyn.setter
    def accel_dyn(self, accel):
        self._accel_dyn = accel

    @property
    def vel(self):
        return self._vel

    @vel.setter
    def vel(self, v):
        self._vel = v

    @property
    def dv(self):
        return self._dv

    @dv.setter
    def dv(self, dv_in):
        self._dv = dv_in

    @property
    def dx(self):
        return self._dx

    @dx.setter
    def dx(self, dx_in):
        self._dx = dx_in

    @property
    def w(self):
        return self.gyro

    @w.setter
    def w(self, w_in):
        self.gyro = w_in

    @property
    def dtheta(self):
        return self._dtheta

    @dtheta.setter
    def dtheta(self, dtheta_in):
        self._dtheta = dtheta_in

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta_in):
        self._theta = theta_in

