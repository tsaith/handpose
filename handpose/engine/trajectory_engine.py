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

        self._image = None # Internal image used to be classified

    def set_ref_quat(self, quat):
        self._ref_projector.q_ref = quat

    def predict_proba(self, vec_arr):
        """
        Predict the type of trajectory.

        data: array
            Trajectory data
        """

        num_vec = len(vec_arr)
        vec_proj = np.zeros_like(vec_arr)
        for i in range(num_vec):
            vec_proj[i] = self._ref_projector.project_vector(vec_arr[i])

        # Vectors on plane
        x_plane = vec_proj[:, 1]
        y_plane = vec_proj[:, 2]

        # Convert the 2D trajectory into an image
        self._image = trajectory_to_image(x_plane, y_plane, broaden_cells=0)

        # Preprocessing
        X = self._image
        X = np.expand_dims(X, axis=2)
        X = np.expand_dims(X, axis=0)
        X = self._classifier.preprocess(X)

        # Predict the probability
        proba = self._classifier.predict_proba(X)

        return proba[0]



