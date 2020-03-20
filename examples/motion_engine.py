import numpy as np
from handpose.vib import has_gesture
from handpose.engine import EngineConfig, MotionEngine

# Fake data
fs = 700
time_duration = 2.0 # Time duration in seconds
num_samples = int(fs * time_duration)

t_indexes = range(num_samples)
theta = np.zeros(num_samples) # Poloidal angle
phi = np.zeros(num_samples)   # Toroidal angle

dtheta = 0.5*np.pi / num_samples
dphi = 0.05*np.pi / num_samples

for i in range(num_samples):
    theta[i] = i*dtheta

# Input features
X_test = np.hstack((theta, phi))
X_test = X_test[np.newaxis, :]

y_test = np.array([1])
y_test = y_test[np.newaxis, :]

# Declare motion engine
config = EngineConfig()
engine = MotionEngine(config)

# Predict classes
y_pred = engine.predict_classes(X_test)


theta_a = theta[0]
theta_z = theta[-1:][0]
phi_a = phi[0]
phi_z = phi[-1:][0]

print("theta_a, theta_z = {}, {}".format(theta_a, theta_z))
print("phi_a, phi_z = {}, {}".format(phi_a, phi_z))
print("Predicted motion classes: {}".format(y_pred))

# Return a motion gesture class
motion_class = y_pred[0]
vib_class = 1
motion_gesture_class = engine.get_MGclass(motion_class, vib_class)
print("When motion_class = {} and vib_class = {},".format(motion_class, vib_class))
print("motion gesture class should be {}".format(motion_gesture_class))

