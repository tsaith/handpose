import numpy as np

class XsensMotionTracker:
    """
    Xsens motion tracker.
    """

    def __init__(self, accel_th=1e-4, vel_th=1e-4, w_th=1e-4, dt=0.01):
        """
        accel_th: float
            Threshold of the dynamic acceleration.
        vel_th: float
            Threshold of velocity
        dt: float
            Time duration in seconds.
        """

        self._accel_th = accel_th
        self._vel_th = vel_th
        self._w_th = w_th
        self._dt = dt
        self._damping = 0.0 # Damping factor
        num_dim = 3 # Number of dimensions

        # Quaternion saved
        self._quat = None

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

        # Unit vector
        self._vec0 = np.array([1.0, 0.0, 0.0])

    def update(self, gyro=None, accel=None, mag=None, accel_dyn=None, motion_status=None, quat=None):
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
        accel_dyn: array
            dynamic acceleration (m/s^2)
        motion_status: int
            motion_status prediced by optical flow.
        quat: Quaternion
            Rotation quaternion. (q_se)
        """
        num_dim = 3

        # Save the quatrnion for tracking
        self.quat = quat

        # Estimate the rotated vector
        vec = quat.rotate_vector(self._vec0)

        # Estimate the position
        self._x = vec

    def get_position():
        return self._x

    def get_velocity():
        return self.vel

    @property
    def quat(self):
        return self._quat

    @quat.setter
    def quat(self, val):
        self._quat = val

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

    @property
    def gyro(self):
        return self.gyro

    @property
    def dtheta(self):
        return self._dtheta

    def get_orientation_angles(self):
        # Return the orientation angles in radain

        quat = self.quat # q_se
        roll = quat.roll
        pitch = quat.pitch
        yaw = quat.yaw

        return roll, pitch, yaw


# -------- Auxiliary functions --------

