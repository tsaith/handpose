
class PPGData:
    """
    Impedance sensor data class.

    """

    def __init__(self, sensor_type):
        """
        Constructor of PPG sensor data.

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


    def set_data(self, ppg_value, timestamp):
        """ 
        Set the PPG value and timestamp.

        Parameters
        ----------
	ppg_value: float
            PPG value.
        timestatmp: long int
            Timestamp.

        """
        pass

    @property
    def data(self):
        """ 
        Return the PPG value and timestamp.

        """
        pass

    @property
    def timestamp(self):
        """ 
        Return the timestamp.

        """
        pass

