
class ImpedanceSensorData:
    """
    Impedance sensor data class.

    """

    def __init__(self, sensor_type):
        """
        Constructor of impedance sensor data.

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


    def set_data(self, imp, timestamp):
        """ 
        Set the impedances and timestamp.

        Parameters
        ----------
	imp: array-like
            Impedances.
        timestatmp: long int
            Timestamp.

        """
        pass

    @property
    def data(self):
        """ 
        Return the impedances and timestamp.

        """
        pass

    @property
    def timestamp(self):
        """ 
        Return the timestamp.

        """
        pass

