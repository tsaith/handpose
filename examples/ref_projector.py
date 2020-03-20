import numpy as np

from handpose.sensor_fusion import *
from handpose.utils import *
from handpose.tracking import *


vec0 = np.array([1, 0, 0])

# Construct a reference projector
projector = RefProjector()

# Make first rotation
theta = 0.0 / 180 * np.pi
phi = 30.0 / 180 *np.pi
q_ref = Quaternion.from_spherical(theta, phi)

# Vector representation in the reference axes
vec_ref = q_ref.rotate_vector(vec0)
projector.q_ref = q_ref # Save the reference quaternion

# Make second rotation
theta = 0.0 / 180 * np.pi
phi = -60.0 / 180 *np.pi
q_target = Quaternion.from_spherical(theta, phi)

# Vector representation in the target axes
vec_target = q_target.rotate_vector(vec0)

# Projected vector
vec_proj = projector.project_vector(vec_target)

print("vec0 = {}".format(vec0))
print("vec_ref = {}".format(vec_ref))
print("vec_target = {}".format(vec_target))
print("vec_proj = {}".format(vec_proj))
