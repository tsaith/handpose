
class JointSensorData:
    """
    Joint sensor data class.

    """

    def __init__(self, sensor_type):
        """
        Constructor of joint sensor data.

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


    def set_data(self, joint_angles, timestamp):
        """ 
        Set the joint angles and timestamp.

        Parameters
        ----------
	joints: array-like
            Joint angles.
        timestatmp: long int
            Timestamp.

        """
        pass

    @property
    def data(self):
        """ 
        Return the joint angles and timestamp.

        """
        pass

    @property
    def timestamp(self):
        """ 
        Return the timestamp.

        """
        pass

