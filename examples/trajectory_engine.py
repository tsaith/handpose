import numpy as np

from handpose.sensor_fusion import *
from handpose.utils import *
from handpose.tracking import *
from handpose.engine import ModelConfig, TrajectoryEngine

# Trajectory engine

dir_path ="../data/symbol/models"
model_file = 'symbol_model.hdf5'
model_path = "{}/{}".format(dir_path, model_file)

config = ModelConfig()
config.model_path = model_path

traj_engine = TrajectoryEngine(config)

# 3D Trajectory
n = 1000
x_arr = np.zeros(n)
y_arr = np.zeros(n)
z_arr = np.zeros(n)
vec_arr = np.zeros((n, 3))

dx = 0.0
dy = 0.1
dz = 0.0

for i in range(n):
    x_arr[i] = i*dx
    y_arr[i] = i*dy
    z_arr[i] = i*dz

    vec_arr[i, 0] = x_arr[i]
    vec_arr[i, 1] = y_arr[i]
    vec_arr[i, 2] = z_arr[i]

# Prepare the reference quaternion
theta = 90.0 / 180 * np.pi
phi = 0.0 / 180 *np.pi
q_ref = Quaternion.from_spherical(theta, phi)

 # Set the reference quaternion
traj_engine.set_ref_quat(q_ref)

# Get the projected vectors
vec_proj_arr = np.zeros_like(vec_arr)
for i in range(len(vec_arr)):
    vec_proj_arr[i] = traj_engine.projected_vector(vec_arr[i])

# Predict the type of trajectory
proba = traj_engine.predict_proba(vec_proj_arr)

print("proba = {}".format(proba))

