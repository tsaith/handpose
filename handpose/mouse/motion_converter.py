import numpy as np
from ..sensor_fusion import *

class MotionConverter:
    """
    Convert the motion information from motion sensor to device.
    """

    def __init__(self, tracker=None, vel_th=1e-5, w_th=1e-2,
                 x_step=1e1, angular_step=1.745e-3):
        """
        Constructor.
    
        Parameters
        ----------
        tracker: Object
            Motion tracker.    
        vel_th: float
            Threshold of velocity normalized to 9.8 (m/s). 
        w_th: float
            Threshold of angular velocity normalized to 2*pi (rad/s).
        x_step: float
            Translational step size.
        angular_step: float
            Angular step size.
        """
        
        self._tracker = tracker
        self._vel_th = vel_th
        self._w_th = w_th
        self._x_step = x_step
        self._angular_step = angular_step

        # Number of dimentions 
        num_dim = 3

        self._vel = np.zeros(num_dim)
        self._dv = np.zeros(num_dim)
        self._dx = np.zeros(num_dim)

        self._w = np.zeros(num_dim) # Angular velocity
        self._dw = np.zeros(num_dim)
        self._dtheta = np.zeros(num_dim)

        # Outputs
        self._dx_out = np.zeros(num_dim)
        self._dtheta_out = np.zeros(num_dim)

    def update(self, gyro, accel, mag):
        """
        Update the status.

        Parameters
        ----------
        gyro: array
            Angular rates (rad/s)
        accel: array
            Measured acceleration (m/s^2)
        mag: array
            Measured magnetic fields (muT)
        """

        tracker = self.tracker

        # Update the tracker
        tracker.update(gyro, accel, mag)

        # Estimate outputs
        self.dx_out = self.estimate_output(self.vel, self.vel_th, self.x_step)
        self.dtheta_out = self.estimate_output(self.w, self.w_th, self.angular_step)
               
    def estimate_output(self, v, v_th, step):

        num_dim = 3
        out = np.zeros(num_dim)

        for i in range(num_dim):
            if np.abs(v[i]) > v_th:
                out[i] = np.sign(v[i])*step
            else:
                out[i] = 0

        return out        

    @property
    def tracker(self):
        return self._tracker

    @property
    def vel(self):
        return self.tracker.vel

    @property
    def vel_th(self):
        return self._vel_th

    @property
    def dv(self):
        return self.tracker.dv

    @property
    def dx(self):
        return self.tracker.dx

    @property
    def dx_out(self):
        return self._dx_out

    @property
    def x_step(self):
        return self._x_step

    @property
    def w(self):
        return self.tracker.w

    @property
    def w_th(self):
        return self._w_th

    @property
    def dw(self):
        return self.tracker.dw

    @property
    def dtheta(self):
        return self.tracker.dtheta

    @property
    def dtheta_out(self):
        return self._dtheta_out

    @property
    def theta(self):
        return self.tracker.theta

    @property
    def angular_step(self):
        return self._angular_step



