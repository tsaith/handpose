import keras

class EngineConfig:
    """
    Engine configuration class.
    """

    def __init__(self):
        """
        Constructor of engine configuration.
        """
        self._model_path = None
        self._scaler_path = None

    def set_model_path(self, model_path):
        """
        Set the model path.
        """
        self._model_path = model_path

        @property
        def model_path(self):
            return self._model_path

        @model_path.setter
        def model_path(self, path):
            self._model_path = path

        @property
        def scaler_path(self):
            return self._scaler_path

        @scaler_path.setter 
        def scaler_path(self, path):
            self._scaler_path = path

