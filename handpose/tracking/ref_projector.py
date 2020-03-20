import numpy as np

class RefProjector:
    """
    Project a vector from the Earth axes into the reference axes.
    """

    def __init__(self):
        """
        Initializer.
        """

        # Reference quaternion (q_s2e)
        self._q_ref = None
        self._vec_proj = None

    def project_vector(self, vec):
        # Project the vector in Earth axes into the reference axes.
        self.vec_proj = self.q_ref.rotate_axes(vec)

        return self.vec_proj

    @property
    def q_ref(self):
        return self._q_ref

    @q_ref.setter
    def q_ref(self, value):
        self._q_ref = value

