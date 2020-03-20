
class AccelerometerData:
    """
    Accelerometer data class.

    """

    def __init__(self, sensor_type):
        """
        Constructor of accelerometer data.

        Parameters
        ----------
        sensor_type: int
            Sensor type.
        """
        pass 

    @property
    def sensor_type(self):
        """ 
        Return the sensor type.
        """
        pass


    def set_data(self, values, timestamp):
        """ 
        Set the values and timestamp.

        Parameters
        ----------
	values: array-like
            Measured values in (x, y, z) format.
        timestatmp: long int
            Timestamp.

        """
        pass

    @property
    def data(self):
        """ 
        Return the measured values and timestamp.

        """
        pass

    @property
    def x(self):
        """ 
        Return the measured x component.

        """
        pass

    @property
    def y(self):
        """ 
        Return the measured y component.

        """
        pass

    @property
    def z(self):
        """ 
        Return the measured z component.

        """
        pass

    @property
    def timestamp(self):
        """ 
        Return the timestamp.

        """
        pass

