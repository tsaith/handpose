import numpy as np
import time
from handpose.sensor_fusion import Quaternion

# Quaternion as numpy array
q_array = np.array([1.0, 0.0, 0.0, 0.0])

# Convert the quaternion as python object
quat = Quaternion.from_array(q_array)

# Define a vector for testing
vec = np.array([1.0, 0.0, 0.0])

# Rotate the vector with quaternion
vec_rot = quat.rotate_axes(vec)

print("quatenion = {}".format(quat.to_array()))
print("Original vector = {}".format(vec))
print("Rotated vector = {}".format(vec_rot))

