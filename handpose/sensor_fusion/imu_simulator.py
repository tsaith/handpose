import numpy as np
from .madgwickahrs import MadgwickAHRS 
from .quaternion import Quaternion

class IMUSimulator:
    """
    IMU Simulator.
    """

    def __init__(self, gyro_e=None, accel_dyn_e=None, q_e2s=None, earth_mag=36.0, earth_dip=30.0):
        """
        Return the raw data of an analytic 9-axis IMU.
    
        Parameters
        ----------
        gyro_e: array
            Angular rate in Earth axes.
        accel_dyn_e: array
            Dynamic acceleration in Earth axes.
        q_e2s: Quaternion
            Rotation quaternion of earth axes respective to sensor axes.
        earth_mag: float
            Magnitude of Earth magnetic field in muT.
        earth_dip: float
            Dip angle of Earth magnetic field.    
        """

        self._gyro_e = gyro_e
        self._accel_dyn_e = accel_dyn_e
        self._q_e2s = q_e2s
        self._earth_mag = earth_mag
        self._earth_dip = earth_dip

    def get_imu_data(self):
        """
        Get the IMU data.
        """

        q_s2e = self.q_e2s.inv()
    
        num_dim = 3
        g_e = np.zeros(num_dim)
        mag_e = np.zeros(num_dim) 
    
        # Gyroscope
        q_tmp = Quaternion.from_vector(self.gyro_e)
        q_tmp = q_s2e*q_tmp*q_s2e.inv()
        gyro = q_tmp.vector
    
        # Acceleration
        g_e[2] = 1.0 # Gravity in Z direction
        accel_e = self.accel_dyn_e + g_e
        q_tmp = Quaternion.from_vector(accel_e)
        q_tmp = q_s2e*q_tmp*q_s2e.inv()
        accel = q_tmp.vector
     
        # Magnetometer
        mag_e[0] = self.earth_mag*np.cos(self.earth_dip) 
        mag_e[1] = 0.0 
        mag_e[2] = -self.earth_mag*np.sin(self.earth_dip) 
        q_tmp = Quaternion.from_vector(mag_e)
        q_tmp = q_s2e*q_tmp*q_s2e.inv()
        mag = q_tmp.vector
    
        return gyro, accel, mag
    
    @property
    def gyro_e(self):
        return self._gyro_e
    
    @property
    def accel_dyn_e(self):
        return self._accel_dyn_e
    
    @property
    def q_e2s(self):
        return self._q_e2s
    
    @property
    def earth_mag(self):
        return self._earth_mag
    
    @property
    def earth_dip(self):
        return self._earth_dip


