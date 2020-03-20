from .sensor_type import SensorType


class Accelerometer:
    """
    Accelerometer class.

    """

    def __init__(self):
        """
        Constructor of accelerometer.

        """
        self.sensor_type = SensorType.ACCELEROMETER

    @property
    def sensor_type(self):
        """ 
        Return the sensor type.
        """
        return self.sensor_type 


    def set_sensor_data(self, sensor_data):
        """ 
        Set the sensor data.

        Parameters
        ----------
	sensor_data: object
            Sensor data.

        """
        pass

    @property
    def sensor_data(self):
        """ 
        Return the sensor data.

        """
        pass

