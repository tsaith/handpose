import numpy as np
from handpose import EngineConfig, GestureEngine

# Scaler and model paths
dir_path = "/home/andrew/projects/handpose/data/mouse/models"
scaler_path = dir_path + "/" + "scaler.dat"
model_path = dir_path + "/" + "mlp_model.hdf"

# Initialize the gesture engine
config = EngineConfig()
config.set_scaler_path(scaler_path)
config.set_model_path(model_path)
engine = GestureEngine(config)

num_features = 28 * 2
X_zeros = np.zeros((1, num_features))

# Make prediction
y_pred = engine.predict_classes(X_zeros)

print("y_pred = {}".format(y_pred))
