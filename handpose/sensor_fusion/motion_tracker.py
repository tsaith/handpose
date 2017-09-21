import numpy as np
from ..sensor_fusion import SensorFusion
from .quaternion import Quaternion
from .motion_classifier import motion_predict

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

        self._quat_saved = None
        self._accel_th = accel_th
        self._vel_th = vel_th
        self._w_th = w_th
        self._dt = dt
        self._beta = beta
        self._num_iter = num_iter

        self._damping = 0.0 # Damping factor

        num_dim = 3 # Number of dimensions

        # Initialize Fusion model
        self._fusion = SensorFusion(self._dt, self._beta,
                                  num_iter=self._num_iter)

        # Translational information
        self._accel_dyn = np.zeros(num_dim) # Dynamic acceleration in sensor axes
        self._accel_dyn_e = np.zeros(num_dim) # Dynamic acceleration in earth axes
        self._accel_dyn_s = np.zeros(num_dim) # Dynamic acceleration in sensor axes
        self._vel = np.zeros(num_dim)
        self._dv = np.zeros(num_dim)
        self._dx = np.zeros(num_dim)
        self._x = np.zeros(num_dim)

        # Angular information
        self._theta = np.zeros(num_dim)  # Relative angles
        self._dtheta = np.zeros(num_dim)
        self._theta_e = np.zeros(num_dim)  # Absolute angles in earth axes

        # IMU information
        self._gyro  = None
        self._accel = None
        self._mag   = None


    def update(self, gyro=None, accel=None, mag=None, video=None):
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
        num_dim = 3

        # Update readings from IMU in sensor axes
        self.gyro = gyro
        self.accel = accel
        self.mag = mag

        if abs(np.linalg.norm(accel) - 1.0) < 1e-3 or self._quat_saved == None:
            # Update the fusion
            if mag is None: # 6-DOF IMU
                self.fusion.update_imu(gyro, accel)
            else: # 9-DOF IMU
                self.fusion.update_ahrs(gyro, accel, mag)
            quat = self.quat
            self._quat_saved = quat
        else:
            quat = self._quat_saved

        # Predict the motion status, 0: static, 1: motional
        self.video = video
        if self.video is None:
            motion_status = 1
        else:
            X = np.expand_dims(video, axis=0)
            motion_status = motion_predict(X)[0]

        # Estimate the dynamic acceleration in Earth axes
        self.accel_dyn = get_dynamic_accel(accel, quat)
        if motion_status == 0:
            self.accel_dyn = np.zeros(num_dim)

        # Filter the noises
        #self.filter_noises(self.accel_dyn, self.accel_th)
        #self.filter_noises(self.vel, self.vel_th)

        # Update velocity and displacement in Earth axes
        self.dv = self.accel_dyn * self.dt
        self.vel +=  self.dv
        #self.vel *= (1.0 - self.damping) # velocity is damping
        if motion_status == 0:
            self.vel = np.zeros(num_dim)

        self.dx = self.vel * self.dt
        self.x += self.dx

        # Update angular velocity and displacement
        self.dtheta = self.w * self.dt
        #for i in range(num_dim): # Filter small noises
        #    if np.abs(self.dtheta[i]) < 5e-5:
        #         self.dtheta[i] = 0.0
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
        q_es = self.quat

        x_s = np.array([1, 0, 0], dtype=np.float64)
        x_se = q_es.rotate_axes(x_s)

        # Project x_e onto the x-y plane in Earth axes
        x_proj = np.array([x_se[0], x_se[1], 0.0])
        norm_proj = np.linalg.norm(x_proj)

        # Poloidal and toroidal angles
        theta = np.arccos(x_se[2]) # Poloidal angle

        phi_t = np.arccos(x_se[0]/norm_proj) # Toroidal angle
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

        q_se = self.quat.inv()

        # z-vector representation of earth axes respective to sensor axes
        z_e = np.array([0, 0, 1], dtype=np.float64)
        z_es = q_se.rotate_axes(z_e)

        # Absolute angles respective to Earth axes
        x_vec = np.array([1, 0, 0])
        y_vec = np.array([0, 1, 0])
        self.theta_e[0] = 0.5*np.pi - np.arccos(np.dot(x_vec, z_es))
        self.theta_e[1] = 0.5*np.pi - np.arccos(np.dot(y_vec, z_es))

    def unit_vectors_s2e(self):
        """
        The representations of unit vectors of sensor axes respective to Earth axes.
        """
        # Rotation quaternion of sensor to Earth axes
        q_es = self.quat
        q_se = q_es.inv()

        # z-vector representation of earth axes respective to sensor axes
        x_e = np.array([1, 0, 0], dtype=np.float64)
        x_es = q_se.rotate_axes(x_e)

        y_e = np.array([0, 1, 0], dtype=np.float64)
        y_es = q_se.rotate_axes(y_e)

        z_e = np.array([0, 0, 1], dtype=np.float64)
        z_es = q_se.rotate_axes(z_e)

        return x_es, y_es, z_es

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

    def get_position():
        return self.x

    def get_velocity():
        return self.vel

    def reset_position(x=None):
        """
        Reset the position.
        """
        if x == None:
            self.x = np.zeros([0, 0, 0], dtype=np.float64)
        else:
            self.x = x

    @property
    def fusion(self):
        return self._fusion

    @fusion.setter
    def fusion(self, fusion_in):
        self.fusion = fusion_in

    @property
    def quat(self):
        # Inverse the q to fit the coordinates used by Madgwick
        return self.fusion.quat.inv()
        #return self.fusion.quat

    @quat.setter
    def quat(self, val):
        self.fusion.quat = val

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
    def video(self):
        return self._video

    @video.setter
    def video(self, value):
        self._video = value

    @property
    def accel_dyn(self):
        return self._accel_dyn

    @accel_dyn.setter
    def accel_dyn(self, accel):
        self._accel_dyn = accel

    @property
    def accel_dyn_e(self):
        return self._accel_dyn_e

    @accel_dyn_e.setter
    def accel_dyn_e(self, accel):
        self._accel_dyn_e = accel

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
    def dv(self, value):
        self._dv = value

    @property
    def dx(self):
        return self._dx

    @dx.setter
    def dx(self, value):
        self._dx = value

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

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

def get_dynamic_accel(accel_s, q_e2s):
    """
    Estimate the dynamic acceleration in Earth axes.

    Paramters
    ---------
    accel_s: array
        Acceleration in sensor axes.
    q_e2s: Quaternion object
        Rotation quaternion, earth axes respective to sensor axes.

    Reurns
    ------
    accel_dyn: array
        Dynamic acceleration in Earth axes.

    """
    accel_e = q_e2s.rotate_axes(accel_s)
    #accel_e = q_e2s.rotate_vector(accel_s)

    # Gravity contribution in the Earth axes,
    # note that the -z direction points down to the Earth
    g_e = np.array([0, 0, 1], dtype=np.float64)

    accel_dyn = accel_e - g_e

    return accel_dyn


def get_dynamic_accel_s(accel_s, q_e2s):
    """
    Estimate the dynamic acceleration in sensor axes.

    Paramters
    ---------
    accel: array
        Acceleration in sensor axes.
    q_e2s: Quaternion object
        Rotation quaternion, earth axes respective to sensor axes.

    Reurns
    ------
    accel_dyn: array
        Dynamic acceleration in sensor axes.

    """
    q_s2e = q_e2s.inv()
    # Gravity contribution in the Earth axes,
    # note that the -z direction points down to the Earth
    g_e = np.array([0, 0, 1], dtype=np.float64)

    g_s = q_s2e.rotate_axes(g_e)
    #g_s = q_s2e.rotate_vector(g_e)

    accel_dyn = accel_s - g_s

    return accel_dyn
