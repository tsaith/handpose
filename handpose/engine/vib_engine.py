import keras
import pickle
from handpose.vib import to_spectrum


# A walk-around to avoid the bug ('module' object has no attribute 'control_flow_ops')\n",
#import keras
#if keras.backend.backend() == 'tensorflow':
#    import tensorflow as tf
#    tf.python.control_flow_ops = tf

class VibEngine:
    """
    Vibrational Gesture engine.
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
        self.config = config
        self.scaler = None
        self.model_trained = None

        # Load the scaler
        self.load_scaler()

        # Load the trained model
        self.load_model()

    def set_config(self, config):
        """
        Set the configuration.

        Parameters
        ----------
        config: object
            Configuration object.

        """
        self.config = config


    @property
    def config(self):
        """
        Return the configuration.
        """
        return self.config

    def load_scaler(self, scaler_path=None):
        """
        Load the scaler file.

        Parameters
        ----------
        scaler_path: str
            The scaler path.
        """

        if scaler_path == None:
            scaler_path = self.config.scaler_path

        self.scaler = pickle.load(open(scaler_path, "rb"))

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

        # Convert to spectrum
        out = to_spectrum(X, keep_dc=False)

        # Scaling
        out = self.scaler.transform(out)

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

    def diagnostic(self):
        """
        Return the diagnostic information.

        """
        pass

