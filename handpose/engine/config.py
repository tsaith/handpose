import keras

class EngineConfig:
    """
    Engine configuration class.
    """

    def __init__(self):
        """
        Constructor of engine configuration.
    
        """
        self.model_path = None
        self.scaler_path = None

    def set_model_path(self, model_path):
        """ 
        Set the model path.
        """
        self.model_path = model_path

    @property 
    def model_path(self):
        """ 
        Return the model path.
        """
        return self.model_path

    def set_scaler_path(self, scaler_path):
        """ 
        Set the scaler path.
        """
        self.scaler_path = scaler_path

    @property 
    def scaler_path(self):
        """ 
        Return the scaler path.
        """
        return self.scaler_path

