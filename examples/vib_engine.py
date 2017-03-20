import numpy as np
from handpose.vib import has_gesture
from handpose.engine import EngineConfig, VibEngine

# Model path
dir_path ="../data/vib/models"
scaler_file = 'scaler.dat'
model_file = 'mlp_model.hdf'

scaler_path = "{}/{}".format(dir_path, scaler_file)
model_path = "{}/{}".format(dir_path, model_file)

# Initialize the gesture engine
config = EngineConfig()
config.set_scaler_path(scaler_path)
config.set_model_path(model_path)
engine = VibEngine(config)

# Testing
fs = 700
num_dims = 3
num_features = fs*num_dims

X_test = np.zeros(num_features)

gesture_exists = has_gesture(X_test)

X_test = X_test[np.newaxis,:]

[proba_pred] = engine.predict_proba(X_test)

print("Has a gesture? {}".format(gesture_exists))
print("Probability information:")
for c in range(len(proba_pred)):
    print("    class: {}, proba: {}".format(c, proba_pred[c]))
