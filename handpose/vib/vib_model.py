import numpy as np
import keras
#import tensorflow as tf
#from tensorflow.contrib import keras
from handpose.vib import to_spectrum, normalize_ts
import pickle


# ---------------------------------------------------------------------

class VibModel:
    """
    Vibration Gesture model.
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
        self._model_trained = None

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

    def load_scaler(self, scaler_path=None):
        """
        Load the scaler file.

        Parameters
        ----------
        scaler_path: str
            The scaler path.
        """

        if scaler_path is None:
            scaler_path = self.config.scaler_path

        if scaler_path is not None:
            self.scaler = pickle.load(open(scaler_path, "rb"), encoding='bytes')

        return self.scaler

    def load_model(self, model_path=None):
        """
        Load the traind model file.

        Parameters
        ----------
        model_path: str
            The model path.
        """
        if model_path == None:
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

        out = X.copy()

        # Convert to spectrum
        #out = to_spectrum(out, keep_dc=False)

        # Normalization
        out = normalize_ts(out)

        # Scaling
        #out = self.scaler.transform(out)

        return out


    def predict_proba(self, X, batch_size=256, verbose=0):
        """
        Return the predicted probabilities..

        Parameters
        ----------
        x: array-like
            Input features.
        """
        X = np.expand_dims(X, axis=0)
        return self.model_trained.predict(X, batch_size=batch_size, verbose=verbose)[0]
        #return self.model_trained.predict_proba(X, batch_size=batch_size, verbose=verbose)[0]


