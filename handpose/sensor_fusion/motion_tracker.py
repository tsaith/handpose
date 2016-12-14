import numpy as np
from ..sensor_fusion import SensorFusion
from .quaternion import Quaternion


class MotionTracker:
    """
    Motion tracker based on sensor fustion.
    """

    def __init__(self, accel_th=0e-3, vel_th=0e-3, w_th=0e-3, dt=0.1, beta=0.041, num_iter=100):
        """
        accel_th: float
            Threshold of the dynamic acceleration.
        vel_th: float
            Threshold of velocity
        dt: float
            Sampling time duration in seconds.
        beta: float
            Parameter used in Madgwick's algorithm.
        num_iter: int
            Number of iterations used to update sensor fusion.
        """

        self._accel_th = accel_th
        self._vel_th = vel_th
        self._w_th = w_th
        self._dt = dt
        self._beta = beta
        self._num_iter = num_iter
        
        self._damping = 0.05 # Damping factor

        num_dim = 3 # Number of dimensions

        # Initialize Fusion model
        self._fusion = SensorFusion(self._dt, self._beta, 
                                  num_iter=self._num_iter)

        # Translational information
        self._accel_dyn = np.zeros(num_dim) # Dynamic acceleration
        self._vel = np.zeros(num_dim)
        self._dv = np.zeros(num_dim)
        self._dx = np.zeros(num_dim)

        # Angular information
        self._theta = np.zeros(num_dim)  # Relative angles
        self._dtheta = np.zeros(num_dim)
        self._theta_e = np.zeros(num_dim)  # Absolute angles in earth axes

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
        self.fusion.update(gyro, accel, mag)

        # Estimate the dynamic acceleration
        self.accel_dyn = estimate_dynamic_accel(accel, self.q) 

        # Filter the noises
        self.filter_noises(self.accel_dyn, self.accel_th)
        self.filter_noises(self.vel, self.vel_th)

        # Update velocity and displacement
        self.dv = self.accel_dyn * self.dt    
        self.vel +=  self.dv
        self.vel *= (1.0 - self.damping) # velocity is damping
        self.dx = self.vel * self.dt

        # Update angular velocity and displacement
        self.dtheta = self.w * self.dt
        for i in range(3): # Filter small noises
            if np.abs(self.dtheta[i]) < 5e-5:
                 self.dtheta[i] = 0.0
        self.theta +=  self.dtheta

        # Estimate the absolute angles respective to Earth axes 
        self.estimate_absolute_angles()

    def filter_noises(self, arr, threshold):
        """ 
        Filter the noises.
        """
        num_dim = 3
        for i in range(num_dim):
            if np.abs(arr[i]) < threshold:
                arr[i] = 0.0

    def estimate_spherical_angles(self):
        """
        Estimate the spherical angles respective to Earth axes.
        """

        # x vector of sensor axes respective to Earth axes
        q_e2s = self.q
        qx_s = Quaternion(0, 1, 0, 0) # x vector in sensor axes
        qx_e = q_e2s*qx_s*q_e2s.inv_unit()
        x_se = qx_e.q[1:]

        # Project x_se onto the x-y plane in Earth axes
        x_proj = np.array([x_se[0], x_se[1], 0.0])
        norm_proj = np.linalg.norm(x_proj)

        # Poloidal and toroidal angles
        theta = np.arccos(x_se[2]) # Poloidal angle

        phi_t = np.arccos(x_se[0]/norm_proj) # Poloidal angle
        if x_se[1] >= 0:
            phi = phi_t
        else:
            phi = -phi_t + 2.0*np.pi

        num_dim = 3
        angles = np.zeros(num_dim)
        angles[0] = theta
        angles[1] = phi

        return angles

    def estimate_absolute_angles(self):
        """
        Estimate the absolute angles respective to Earth axes.
        """
        num_dim = 3

        # z-vector representation of earth axes respective to sensor axes 
        q_s2e = self.q.inv_unit()
        qz_e = Quaternion(0, 0, 0, 1) # z vector in earth axes
        qz_s = q_s2e*qz_e*q_s2e.inv_unit()
        z_es = qz_s.q[1:]

        # Absolute angles respective to Earth axes
        x_vec = np.array([1, 0, 0])
        y_vec = np.array([0, 1, 0])
        self.theta_e[0] = 0.5*np.pi - np.arccos(np.dot(x_vec, z_es))
        self.theta_e[1] = 0.5*np.pi - np.arccos(np.dot(y_vec, z_es))
        self.theta_e[2] = 0.0 # This has not been estimated yet.

    def init_motion_data(self):
        """
        Initialize the motion data.
        """
        num_dim = 3

        # Translational information
        self.accel_dyn = np.zeros(num_dim)
        self.vel = np.zeros(num_dim)

        # Angular information
        self.theta = np.zeros(num_dim)

    @property
    def fusion(self):
        return self._fusion

    @fusion.setter
    def fusion(self, fusion_in):
        self.fusion = fusion_in

    @property
    def q(self):
        return self.fusion.q

    @property
    def accel_th(self):
        return self._accel_th

    @property
    def vel_th(self):
        return self._vel_th

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

    @property
    def theta_e(self):
        return self._theta_e

    @theta.setter
    def theta_e(self, theta_in):
        self._theta_e = theta_in

    @property
    def damping(self):
        return self._damping

    @damping.setter
    def damping(self, value):
        self._damping = value

def estimate_dynamic_accel(accel, q_e2s):
    """
    Estimate the dynamic acceleration.
    
    Paramters
    ---------
    accel: array
        Acceleration in sensor axes.
    q_e2s: Quaternion object
        Rotation quaternion, earth axes to sensor axes.
        
    Reurns
    ------
    accel_dyn: array
        Dynamic acceleration in sensor axes.
    
    """
    q_s2e = q_e2s.inv_unit() # Sensor axes to Earth axes 
    # Gravity contribution in the Earth axes, 
    # please note that the accelerometer feels 1g in z direction
    # when the sensor is static 
    g_e = Quaternion(0, 0, 0, 1)
    g_s = q_s2e*g_e*q_s2e.inv_unit() # Gravity in the sensor axes
    
    accel_dyn = accel - g_s.vector

    return accel_dyn 


