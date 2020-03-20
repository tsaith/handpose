from enum import Enum

class SensorType(Enum):
    """
    Enumeration of sensor type.

    Attributes
    ----------

    IMPEDANCE_SENSOR: int
        Type of impedance sensor

    JOINT_SENSOR: int
        Type of joint sensor

    ACCELEROMETER: int
        Type of accelerometer

    GYROSCOPE: int
        Type of gyroscope

    MAGNETOMETER: int
        Type of magnetometer

    PPG: int
        Type of PPG

    """

    IMPEDANCE_SENSOR = 0
    JOINT_SENSOR = 1
    ACCELEROMETER = 2
    GYROSCOPE = 3
    MAGNETOMETER = 4
    PPG = 5

