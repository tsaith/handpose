
class DataGlove:
    """
    Data glove class.

    """

    def __init__(self, joint_sensor, accelerometer, gyroscope, 
        magnetometer, ppg):
        """
        Constructor of wristband.

        Parameters
        ----------
	joint_sensor: object
            Joint sensor
	accelerometer: object
            Accelerometer
	gyroscope: object
            Gyroscope
	magnetometer: object
            Magnetometer
        ppg: object
            PPG
        """
        pass 


    def set_joint_sensor(self, joint_sensor):
        """ 
        Set the joint sensor.

        Parameters
        ----------
	joint_sensor: object
            Joint sensor.

        """
        pass

    @property
    def joint_sensor(self):
        """ 
        Return the joint sensor.

        """
        pass

    def set_accelerometer(self, accelerometer):
        """ 
        Set the accelerometer.

        Parameters
        ----------
	accelerometer: object
            Accelerometer.

        """
        pass

    @property
    def accelerometer(self):
        """ 
        Return the accelerometer.

        """
        pass

    def set_gyroscope(self, gyroscope):
        """ 
        Set the gyroscope.

        Parameters
        ----------
	gyroscope: object
            Gyroscope.

        """
        pass

    @property
    def gyroscope(self):
        """ 
        Return the gyroscope.

        """
        pass

    def set_magnetometer(self, magnetometer):
        """ 
        Set the magnetometer.

        Parameters
        ----------
	magnetometer: object
            Magnetometer.

        """
        pass

    @property
    def magnetometer(self):
        """ 
        Return the magnetometer.

        """
        pass

