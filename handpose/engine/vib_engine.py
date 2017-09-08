import numpy as np
import tensorflow as tf
import tensorlayer as tl

import tensorflow.contrib.keras.api.keras as keras
import pickle
from handpose.vib import to_spectrum, normalize


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
        self._config = config
        self._scaler = None
        self._batch_size = None

        tf.reset_default_graph()
        self._sess = tf.InteractiveSession()

        # Retrive the ops
        self._x_holder = None
        self._proba_op = None
        self._label_op = None

        # Load the trained model
        self.load_model()

        # Initializer will initialize the variables imported
        #self.sess.run(tf.global_variables_initializer())

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
    def sess(self):
        return self._sess

    @sess.setter
    def sess(self, val):
        self._sess = val

    @property
    def model_trained(self):
        return self._model_trained

    @model_trained.setter
    def model_trained(self, val):
        self._model_trained= val

    def load_model(self):
        """
        Load the traind model file.

        Parameters
        ----------
        model_path: str
            The model path.
        """
        graph_path = self.config.graph_path
        model_path = self.config.model_path

        # Import the meta graph
        tf.train.import_meta_graph(graph_path, clear_devices=True)

        # Import weights
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)

        # Retrive the ops
        self._x_holder = tf.get_collection('x_holder')[0]
        self._proba_op = tf.get_collection('proba_op_t')[0]
        self._label_op = tf.get_collection('label_op_t')[0]

        return None

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

        # Add the channel
        if out.ndim < 3:
            out = np.expand_dims(out, axis=2)

        return out

    def predict_classes(self, X, batch_size=256, verbose=0):
        """
        Generate class predictions for the input samples.

        Parameters
        ----------
        X: array-like
            Input features.
        """
        return self.sess.run(self._label_op, feed_dict={self._x_holder: X})


    def predict_proba(self, X, batch_size=None, verbose=0):
        """
        Return the predicted probabilities..

        Parameters
        ----------
        x: array-like
            Input features.
        """
        return self.sess.run(self._proba_op, feed_dict={self._x_holder: X})

    def predict(self, X, batch_size=None, verbose=0):
        """
        Generates output predictions for the input samples.

        Parameters
        ----------
        x: array-like
            Input features.
        """
        return self.sess.run(self._proba_op, feed_dict={self._x_holder: X})

    def diagnostic(self):
        """
        Return the diagnostic information.

        """
        pass

# ---------------------------------------------------------------------

class VibEngineKeras:
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
        self._config = config
        self._scaler = None
        self._model_trained = None

        # Load the scaler
        self.load_scaler()

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

        if scaler_path == None:
            scaler_path = self.config.scaler_path

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

