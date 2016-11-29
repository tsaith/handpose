import numpy as np
from ..sensor_fusion import SensorFusion
from .quaternion import Quaternion


class MotionTracker:
    """
    Motion tracker based on sensor fustion.
    """

    def __init__(self, accel_th=0e-4, dt=0.1, beta=0.041, num_iter=100):
        """
        accel_th: float
            Threshold of the dynamic acceleration.
        dt: float
            Sampling time duration in seconds.
        beta: float
            Parameter used in Madgwick's algorithm.
        num_iter: int
            Number of iterations used to update sensor fusion.
        """

        self._accel_th = accel_th
        self._dt = dt
        self._beta = beta
        self._num_iter = num_iter
        
        num_dim = 3 # Number of dimensions

        # Initialize Fusion model
        self._fuse = SensorFusion(self._dt, self._beta, 
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

        # Update the fusion
        self.fuse.update(gyro, accel, mag)

        # Estimate the dynamic acceleration
        self.accel_dyn = estimate_dynamic_accel(accel, self.q) 

        # Filter the noises in dynamic acceleration 
        self.filter_accel_dyn(self.accel_dyn, self.accel_th)

        # Update velocity and displacement
        self.dv = self.accel_dyn * self.dt    
        self.vel +=  self.dv
        self.dx = self.vel * self.dt

        # Update angular velocity and displacement
        self.dtheta = self.w * self.dt
        self.theta +=  self.dtheta


    def filter_accel_dyn(self, accel, accel_th):
        """ 
        Filter the noises in dynamic acceleration.
        """
        num_dim = 3
        for i in range(num_dim):
            if np.abs(accel[i]) < accel_th:
                accel[i] = 0.0

    @property
    def fuse(self):
        return self._fuse

    @property
    def q(self):
        return self.fuse.quaternion

    @property
    def accel_th(self):
        return self._accel_th

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


def estimate_dynamic_accel(accel, q_e2s):
    """
    Estimate the dynamic acceleration.
    
    Paramters
    ---------
    accel: array
        Acceleration in sensor axes.
    q_e2s: Quaternion object
        Rotation quaternion denoting earth axes to sensor axes.
        
    Reurns
    ------
    accel_dyn: array
        Dynamic acceleration in sensor axes.
    
    """
    q_s2e = q_e2s.inv_unit() # Sensor axes to Earth axes 
    g_e = Quaternion(0, 0, 0, 1) # Gravity in the Earth axes
    g_s = q_s2e*g_e*q_s2e.inv_unit() # Gravity in the sensor axes
    
    accel_dyn = accel - g_s.vector

    return accel_dyn 

def gravity_compensate(accel, q):
    """
    Compensate the accelerometer readings from gravity. 
    
    Parameters
    ----------
    q: array
        The quaternion representing the orientation of a 9DOM MARG sensor array
    acc: array
        The readings coming from an accelerometer expressed in g
        
    Returns
    -------
        acc_out: array
            a 3d vector representing dynamic acceleration expressed in g
    """
    num_dim = 3
    g = np.zeros(num_dim)
    accel_out = np.zeros(num_dim)
  
    # Get expected direction of gravity
    g[0] = 2 * (q[1] * q[3] - q[0] * q[2])
    g[1] = 2 * (q[0] * q[1] + q[2] * q[3])
    g[2] = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]
  
    # Compensate accelerometer readings with the expected direction of gravity
    accel_out[:] = accel[:] - g[:]
    
    return accel_out

