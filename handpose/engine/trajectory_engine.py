import numpy as np
import tensorflow as tf
from tensorflow.contrib import keras
import pickle


# ---------------------------------------------------------------------

class TrajectoryEngine:
    """
    Trajectory engine.
    """

    def __init__(self, config):
        """
        Constructor.

        Parameters
        ----------
        config : object
            Configuration containing the information of system,
            pre-processing and post-processing.

        """
        self._config = config
        self._scaler = None
        self._model_trained = None # Symbol model

        # Load the scaler
        #self.load_scaler()

        # Load the trained model
        self.load_model()

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, val):
        self._config = val

    @property
    def scaler(self):
        return self._scaler

    @scaler.setter
    def scaler(self, val):
        self._scaler = val

    @property
    def model_trained(self):
        return self._model_trained

    @model_trained.setter
    def model_trained(self, val):
        self._model_trained= val

    def load_scaler(self):
        """
        Load the scaler file.

        Parameters
        ----------
        scaler_path: str
            The scaler path.
        """
        scaler_path = self.config.scaler_path
        self.scaler = pickle.load(open(scaler_path, "rb"), encoding='bytes')

        return self.scaler

    def load_model(self):
        """
        Load the traind model file.

        Parameters
        ----------
        model_path: str
            The model path.
        """
        model_path = self.config.model_path
        self.model_trained = keras.models.load_model(model_path)

        return self.model_trained

    def preprocess(self, X):
        """
        Preprocess the input features

        Parameters
        ----------
        scaler_path: str
            The scaler path.
        """

        # Convert to spectrum
        out = to_spectrum(X, keep_dc=False)

        # Normalization
        out = normalize(out)

        # Scaling
        #out = self.scaler.transform(out)

        # Add the channel
        if out.ndim < 3:
            out = np.expand_dims(out, axis=2)

        return out

    def predict_classes(self, x, batch_size=256, verbose=0):
        """
        Generate class predictions for the input samples.

        Parameters
        ----------
        X: array-like
            Input features.
        """

        return self.model_trained.predict_classes(x, batch_size=batch_size, verbose=verbose)



    def predict_proba(self, x, batch_size=256, verbose=0):
        """
        Return the predicted probabilities..

        Parameters
        ----------
        x: array-like
            Input features.
        """
        return self.model_trained.predict_proba(x, batch_size=batch_size, verbose=verbose)

    def predict(self, x, batch_size=256, verbose=0):
        """
        Generates output predictions for the input samples.

        Parameters
        ----------
        x: array-like
            Input features.
        """
        return self.model_trained.predict(x, batch_size=batch_size, verbose=verbose)


