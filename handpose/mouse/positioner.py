import numpy as np
from ..sensor_fusion import *

class CursorPositioner:
    """
    Cursor positioner which estimates the movement of cursor.
    """

    def __init__(self, tracker=None, vel_th=1e-3, step_size=10, x_scaling=1e4):
        """
        Constructor.
    
        Parameters
        ----------
        tracker: Object
            Motion tracker.    
        """
        
        self._tracker = tracker
        self._vel_th = vel_th # Velocity threshold
        self._step_size = step_size # Step size to move in pixels
        self._x_scaling = x_scaling # Scaling factor

        num_dim = 3
        self._vel = np.zeros(num_dim)
        self._dv = np.zeros(num_dim)
        self._dx = np.zeros(num_dim)


        # Output used to control the cursor
        self._dx_out = np.zeros(num_dim, dtype=np.int32)

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

        # Update velocity and displacement
        self.vel = tracker.vel
        self.dv = tracker.dv
        self.dx = tracker.dx

        # Estimate the output dx
        self.estimate_dx_out(self.vel, self.vel_th)
               
    def estimate_dx_out(self, vel, vel_th):

        num_dim = 3
        for i in range(num_dim):
            if np.abs(vel[i]) > vel_th:
                self._dx_out[i] = np.sign(vel[i])*self.step_size
            else:
                self._dx_out[i] = 0

    @property
    def tracker(self):
        return self._tracker

    @property
    def vel(self):
        return self._vel

    @property
    def vel_th(self):
        return self._vel_th

    @property
    def dv(self):
        return self._dv

    @property
    def dx(self):
        return self._dx

    @property
    def dx_out(self):
        return self._dx_out

    @property
    def step_size(self):
        return self._step_size




