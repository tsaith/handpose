import numpy as np

class MPU9250:

    def __init__(self):
        """
        Constuctor of MPU9250.
        """

        num_dim = 3 # Number of dimensions

        # Readings in the sensor axes
        self._gyro  = np.zeros(num_dim)
        self._accel = np.zeros(num_dim)
        self._mag   = np.zeros(num_dim)

    def convert_axes(self, gyro, accel, mag):
        """
        Convert to the sensor axes.

        Parameters
        ----------
        gyro: array
            Angular rates from the gyroscope.
        accel: array
            Acceleration from the accelerometer.
        mag: array
            Magnetic fields from the magnetometer.

        Returns
        -------
        imu: list
            Readings of IMU in the sensor axes.
        """

        self.gyro[0] =  gyro[0]
        self.gyro[1] = -gyro[1]
        self.gyro[2] = -gyro[2]

        self.accel[0] =  accel[0]
        self.accel[1] = -accel[1]
        self.accel[2] = -accel[2]

        self.mag[0] =  mag[1]
        self.mag[1] = -mag[0]
        self.mag[2] =  mag[2]

        imu = [self.gyro, self.accel, self.mag]

        return imu
       
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
        

