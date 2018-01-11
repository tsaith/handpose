import numpy as np
import tensorflow as tf
from tensorflow.contrib import keras
import pickle

from handpose.tracking import *
from handpose.utils import *

# ---------------------------------------------------------------------

class TrajectoryEngine:
    """
    Trajectory engine which predicts the type of the 3D trajectory.
    """

    def __init__(self, config):
        """
        Constructor.

        Parameters
        ----------
        config : object
            Configuration containing the information of system,
            pre-processing and post-processing.
        dt: float
            Time duration of motion sensor.
        """
        self._ref_projector = RefProjector()
        self._classifier = SymbolClassifier(config)

        # Position projected on the reference plane.
        self._x_plane = None
        self._y_plane = None

        # Internal image used to be classified
        self._image = None

    def set_ref_quat(self, quat):
        self._ref_projector.q_ref = quat

    def projected_vector(self, vec):
        """
        Return the projected vector on the reference plane.

        vec: array
            3D position vector.
        """
        return self._ref_projector.project_vector(vec)

    def predict_proba(self, vec_proj):
        """
        Predict the type of trajectory.

        vec_proj: array
            Projected vectors.
        """

        # Convert the (y, z) of projected vectors into an image
        self._image = trajectory_to_image(vec_proj[:, 1], vec_proj[:, 2], broaden_cells=0)

        # Preprocessing
        X = self._image
        X = np.expand_dims(X, axis=2)
        X = np.expand_dims(X, axis=0)
        X = self._classifier.preprocess(X)

        # Predict the probability
        proba = self._classifier.predict_proba(X)

        return proba[0]



