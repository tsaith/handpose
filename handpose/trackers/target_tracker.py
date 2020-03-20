# import the necessary packages

import sys
import os
import numpy as np

from .centroid_tracker import CentroidTracker
from .lib import *


class TargetTracker():

    def __init__(self, frame_width=None, frame_height=None, num_disappered_max=20):
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.position = None

        self.ct = CentroidTracker(maxDisappeared=num_disappered_max)

    def update(self, frame):

        faces = detect_faces(frame)

        self.ct.update(faces)
        self.position = self.ct.get_target_position()

    def get_position(self):
        return self.position

    def get_position_norm(self):

        p = None
        if self.position is not None:
            x, y = self.position

            x = float(x) / self.frame_width
            y = float(y) / self.frame_height
            p = (x, y)

        return p
