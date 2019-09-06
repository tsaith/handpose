from handpose.sensors import SensorType


class ImpSensorData:
    """
    Impedance sensor data class.

    """

    def __init__(self):
        """
        Constructor of impedance sensor data.

        """
        self.sensor_type = SensorType.IMPEDANCE_SENSOR
        self.data = None
        self.timestamp = None

    @property
    def sensor_type(self):
        """ 
        Return the sensor type.
        """
        return sensor_type


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
        self.data = imp
        self.timestamp = timestamp

    @property
    def data(self):
        """ 
        Return the impedances and timestamp.

        """
        return self.data

    @property
    def timestamp(self):
        """ 
        Return the timestamp.

        """
        return timestamp

