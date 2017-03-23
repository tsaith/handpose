import numpy as np
import keras
import pickle


# A walk-around to avoid the bug ('module' object has no attribute 'control_flow_ops')\n",
import keras
if keras.backend.backend() == 'tensorflow':
    import tensorflow as tf
    tf.python.control_flow_ops = tf


def get_motion_class(theta, phi):
    """
    Return the motion class.

    Parameters
    ----------
    theta: array-like
        Poloidal angles.
    phi: array-like
        Toroidal angles.

    Returns
    -------
    motion_class: int
        Motion class, where
        0, 1, 2, 3, 4 -> static, up, down, left, right
    """

    angle_c = np.pi / 18.0 # Critical angle

    theta_a = theta[0]
    theta_z = theta[-1:][0]
    phi_a = phi[0]
    phi_z = phi[-1:][0]

    motion_class = 0 # static as default

    is_vertical = False
    is_horizontal = False

    if np.abs(theta_z - theta_a) > angle_c: # Vertical
        is_vertical = True
        if theta_z > theta_a: # Down
            motion_class = 2
        else: # Up
            motion_class = 1

    else: # Not vertical

        if np.abs(phi_z - phi_a) > angle_c: # Horizontal
            is_horizontal = True
            if phi_z > phi_a: # left
                motion_class = 3
            else: # right
                motion_class = 4

    return motion_class


def predict_classes_analytic(X):

    num_samples, num_features = X.shape

    num_half = num_features / 2

    classes = []

    for i in range(num_samples):
        theta = X[i, :num_half]
        phi = X[i, num_half:]

        # Determine and store the motion class
        classes.append(get_motion_class(theta, phi))

    return classes


class MotionEngine:
    """
    Motion engine.
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
        #self.load_scaler()

        # Load the trained model
        #self.load_model()

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


    def predict_classes(self, x, batch_size=256, verbose=0):
        """
        Generate class predictions for the input samples.

        Parameters
        ----------
        X: array-like
            Input features.
        """
        #return self.model_trained.predict_classes(x, batch_size=batch_size, verbose=verbose)
        return predict_classes_analytic(x)



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

    def get_MGclass(self, motion_class, vib_class):
        """
        Return the class of motion gesture.

        motion_class: int
            Motion trajectory class.
            0, 1, 2, 3, 4 -> static, up, down, left, right
        vib_class: int
            Vibrational gesture class.
            0, 1 -> snap, flick
        """

        return (motion_class+1)*(vib_class+1) - 1

    def diagnostic(self):
        """
        Return the diagnostic information.

        """
        pass

