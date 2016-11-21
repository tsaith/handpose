import numpy as np
from .madgwickahrs import MadgwickAHRS 
from .quaternion import Quaternion

def dynamic_accel(accel, q_e2s):
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
        q = fuse.quaternion
        qs.append(q)

    return fuse, qs     


def analytic_imu_data(sensor_angles, earth_mag=36.0, earth_dip=30.0):
    """
    Return the raw data of an analytic 9-axis IMU.

    Parameters
    ----------
    sensor_angles: array-like
        Sensor orientaion angles represented as roll, pitch and jaw in unit of radian.
    earth_mag: float
        Magnitude of Earth magnetic field in muT.
    earth_dip: float
        Dip angle of Earth magnetic field.    
   
    Returns
    -------
    imu: list
        IMU data which contains gyro, accel and mag.
    """

    roll, pitch, jaw = sensor_angles
                                              
    num_dims = 3
    gyro = np.zeros(num_dims) 
    accel = np.zeros(num_dims) 
    mag = np.zeros(num_dims) 
    
    accel[0] = np.cos(0.5*np.pi + pitch) 
    accel[1] = 0 
    accel[2] = np.sin(0.5*np.pi + pitch) 
    
    mag[0] = earth_mag*np.cos(earth_dip + pitch) 
    mag[1] = 0.0 
    mag[2] = earth_mag*np.sin(earth_dip + pitch) 
    
    imu = [gyro, accel, mag]

    return imu


