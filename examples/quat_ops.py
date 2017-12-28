"""
Operations of quaternion.
"""

import numpy as np
import time
from handpose.utils import Quaternion

# Quaternion as numpy array
angle = 0.5*np.pi # Angle for rotation
c = np.cos(0.5*angle)
s = np.sin(0.5*angle)
q_array = np.array([c, 0.0, s, 0.0]) # Rotate about the y axis

# Convert the quaternion as python object
quat = Quaternion.from_array(q_array)

# Define a vector for testing
vec_ori = np.array([1.0, 0.0, 0.0])

# Rotate the vector with quaternion
vec_rot = quat.rotate_axes(vec_ori)

print("quatenion = {}".format(quat.to_array()))
print("Original vector = {}".format(vec_ori))
print("Rotated vector = {}".format(vec_rot))

