import numpy as np
from .madgwickahrs import MadgwickAHRS 
from .quaternion import Quaternion

class SensorFusion:
    """
    Sensor fusion.
    """

    def __init__(self, dt=0.1, beta=0.41, num_iter=100):
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

    def update(self, gyro, accel, mag):
        """
        Update of fusion.
        """

        qs = []
        for i in range(self.num_iter):
            self.fuse.update(gyro, accel, mag)
            qs.append(self.fuse.quaternion)
        
        return qs     


    @property
    def fuse(self):
        return self._fuse
    
    @fuse.setter
    def fuse(self, fuse_in):
        return self._fuse

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
    def quaternion(self):
        return self.fuse.quaternion


def init_madgwick(dt, beta, num_iter=100):
    
    sub_dt = dt / num_iter
    q0 = Quaternion([1, 0, 0, 0])
    fuse = MadgwickAHRS(sampleperiod=sub_dt, 
                        quaternion=q0, beta=beta)

    return fuse    


def fusion_update(fuse, imu, num_iter=100):
    
    gyro, accel, mag = imu
    qs = []
    for i in range(num_iter):
        fuse.update(gyro, accel, mag)
        qs.append(fuse.quaternion)

    return fuse, qs     


def analytic_imu_data(gyro_e, accel_dyn_e, q_e2s, earth_mag=36.0, earth_dip=30.0):
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
   
    Returns
    -------
    imu: list
        IMU data which contains gyro, accel and mag.
    """

    q_s2e = q_e2s.inv()

    num_dim = 3
    g_e = np.zeros(num_dim)
    mag_e = np.zeros(num_dim) 

    # Gyroscope
    q_tmp = Quaternion.from_vector(gyro_e)
    q_tmp = q_s2e*q_tmp*q_s2e.inv()
    gyro = q_tmp.vector

    # Acceleration
    g_e[2] = 1.0 # Gravity in Z direction
    accel_e = accel_dyn_e + g_e
    q_tmp = Quaternion.from_vector(accel_e)
    q_tmp = q_s2e*q_tmp*q_s2e.inv()
    accel = q_tmp.vector
 
    # Magnetometer
    mag_e[0] = earth_mag*np.cos(earth_dip) 
    mag_e[1] = 0.0 
    mag_e[2] = earth_mag*np.sin(earth_dip) 
    q_tmp = Quaternion.from_vector(mag_e)
    q_tmp = q_s2e*q_tmp*q_s2e.inv()
    mag = q_tmp.vector

    # Combine three sensors
    imu = [gyro, accel, mag]

    return imu


