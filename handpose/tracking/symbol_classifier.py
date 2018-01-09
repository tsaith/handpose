import numpy as np
import tensorflow as tf
from tensorflow.contrib import keras
import pickle


# ---------------------------------------------------------------------

class SymbolClassifier:
    """
    Symbol classifier.
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
        self._model_trained = None

        # Load the trained model
        self.load_model()

    @property
    def config(self):
        return self._config

    @property
    def scaler(self):
        return self._scaler

    @property
    def model_trained(self):
        return self._model_trained

    def load_model(self):
        """
        Load the traind model file.

        Parameters
        ----------
        model_path: str
            The model path.
        """
        model_path = self.config.model_path
        self._model_trained = keras.models.load_model(model_path)

        return self._model_trained

    def preprocess(self, X):
        """
        Preprocess the input features

        Parameters
        ----------
        scaler_path: str
            The scaler path.
        """

        # Normalization
        out = X.astype('float32')
        out /= 255

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


