import keras

class ModelConfig:
    """
    Engine configuration class.
    """

    def __init__(self):
        """
        Constructor of engine configuration.
        """
        self._model_path = None
        self._graph_path = None
        self._scaler_path = None


    @property
    def model_path(self):
        return self._model_path

    @model_path.setter
    def model_path(self, path):
        self._model_path = path

    @property
    def graph_path(self):
        return self._graph_path

    @graph_path.setter
    def graph_path(self, path):
        self._graph_path = path

    @property
    def scaler_path(self):
        return self._scaler_path

    @scaler_path.setter
    def scaler_path(self, path):
        self._scaler_path = path

# ---------

class EngineConfig(ModelConfig):
    """
    Engine configuration class.
    """

    def __init__(self):
        """
        Constructor of engine configuration.
        """
        super().__init__()

