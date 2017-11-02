import numpy as np
#from .python_quaternion import Quaternion
#from .fast_quaternion import Quaternion

class IMUSimulator:
    """
    IMU Simulator.
    """

    def __init__(self, gyro_u=None, accel_dyn_u=None, q_se=None, earth_mag=36.0, earth_dip=30.0):
        """
        Return the raw data of an analytic 9-axis IMU.

        Parameters
        ----------
        gyro_u: array
            Angular rate in Earth axes.
        accel_dyn_u: array
            Dynamic acceleration in Earth axes.
        q_se: Quaternion
            Rotation quaternion of sensor axes respective user coordinates.
        earth_mag: float
            Magnitude of Earth magnetic field in muT.
        earth_dip: float
            Dip angle of Earth magnetic field.
        """

        self._gyro_u = gyro_u
        self._accel_dyn_u = accel_dyn_u
        self._q_se = q_se
        self._earth_mag = earth_mag
        self._earth_dip = earth_dip

    def get_imu_data(self):
        """
        Get the IMU data.

        User coordinates are employed.
        """

        num_dim = 3
        g_e = np.zeros(num_dim)
        mag_e = np.zeros(num_dim)

        # Gyroscope
        gyro = self.q_se.rotate_axes(self.gyro_u)

        # Acceleration
        g_e[2] = 1.0 # Gravity contribution in +Z direction
        accel_e = self.accel_dyn_u + g_e
        accel = self.q_se.rotate_axes(accel_e)

        # Magnetometer
        mag_e[0] = self.earth_mag*np.cos(self.earth_dip)
        mag_e[1] = 0.0
        mag_e[2] = -self.earth_mag*np.sin(self.earth_dip)
        mag = self.q_se.rotate_axes(mag_e)

        return gyro, accel, mag

    @property
    def gyro_u(self):
        return self._gyro_u

    @property
    def accel_dyn_u(self):
        return self._accel_dyn_u

    @property
    def q_se(self):
        return self._q_se

    @property
    def earth_mag(self):
        return self._earth_mag

    @property
    def earth_dip(self):
        return self._earth_dip


