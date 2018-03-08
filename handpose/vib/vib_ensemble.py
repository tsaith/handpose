import numpy as np
from ..utils import to_magnitude

class VibEnsemble:
    """
    Vibration ensemble.
    """

    def __init__(self, sampling_rate, num_preds, model):

        self._sampling_rate = sampling_rate
        self._num_preds = num_preds
        self._model = model

        self._period_a = int(0.6*sampling_rate) # Period 1
        self._period_b = int(0.2*sampling_rate) # Period 2

        # Data buffer
        accel_init = [0.0, 0.0, 1.0] # Magnitude is one
        self._buffer = [np.asarray(accel_init) for i in range(sampling_rate)]

        self._detect = True
        self._has_gesture = False
        self._counter = 0 # Data counter
        self._probas = [] # # Probability distribution of all predictions

    def update(self, data, threshold=0.15):
        """
        Update recognition.

        Parameters
        ----------
        data: array-lie
            Input data.
        threshold: float
            Signal threshold.

        Returns
        -------
        outputs: array-like
            Output results. It is None when the process is under going and
            returns reduced probability when the process is done.
        """


        self._buffer.append(data) # Append new data into the buffer
        self._buffer.pop(0) # Remove the fist item.

        # Perform detection
        if self._detect:
            accel = data[0:3]
            diff_a = to_magnitude(accel) - 1.0

            # Has a gesture?
            if diff_a > threshold:
                self._has_gesture = True

        # When a gesture might exist
        if self._has_gesture:
            self._counter += 1

            if (self._counter == self._period_a) or \
               (self._counter > self._period_a and self._counter % self._period_b == 0):
                # Prepare the feature
                X = self._buffer[-self._sampling_rate:]
                X = np.asarray(X)
                X = self._model.preprocess(X)
                proba = self._model.predict_proba(X)

                # Save the probabilities
                self._probas.append(proba)

        # Output the probability distribution
        pd = None
        if len(self._probas) == self._num_preds:
            pd = np.asarray(self._probas)

            # Finalization
            self._probas = []
            self._detect = True
            self._has_gesture = False
            self._counter = 0

        return pd
