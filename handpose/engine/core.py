
class GestureEngine:
    """
    Gesture engine class.

    """

    def __init__(self, config):
        """
        Constructor of recognition engine.
    
        Parameters
        ----------
        config : object
            Configuration containing the information of system, 
            pre-processing and post-processing.

        """
        pass 



    def set_config(self, config):
        """ 
        Set the configuration.

        Parameters
        ----------
	config: object
            Configuration object.

        """
        pass

    
    @property
    def config(self):
        """
        Return the configuration.
        """
        pass

    def set_wristband(self, wristband):
        """
        Set the wristband.

        Parameters
        ----------
	wristband: object
            Wristband object.
        """
        pass

    @property
    def wristband(self):
        """
        Return the wristband.
        """
        pass

    def set_glove(self, glove):
        """
        Set the glove.

        Parameters
        ----------
	glove: object
            glove object.
        """
        pass

    @property
    def glove(self):
        """
        Return the glove.
        """
        pass

    def predict(self, network_type, args):
        """
        Return the predicted hand model.

        Parameters
        ----------
	network_type: int
            Network type
	args: array-like 
            Arguments used in the netwrok
        """
        pass

    def diagnostic(self):
        """
        Return the diagnostic information.

        """
        pass



class EngineConfig:
    """
    Engine configuration class.

    """

    def __init__(self):
        """
        Constructor of engine configuration.
    
        """
        pass 



    def set_dt(self, dt):
        """ 
        Set the time duration.

        Parameters
        ----------
	dt: float
            Time duration in milliseconds.

        """
        pass

