import numpy as np
from handpose.vib import has_gesture
from handpose.engine import EngineConfig, VibEngine
from handpose.utils import set_cuda_visible_devices

# Device configuration
set_cuda_visible_devices("") # CPU only

# Model path
dir_path ="../data/vib/models"

model_file = 'trained_model.hdf5'

model_path = "{}/{}".format(dir_path, model_file)

# Initialize the gesture engine
config = EngineConfig()
config.model_path = model_path
engine = VibEngine(config)


# Testing
fs = 700
num_dims = 3
num_features = fs*num_dims

X_test = np.ones(num_features)

# Preprocess the features
X_test = np.expand_dims(X_test, axis=0)
X_test = engine.preprocess(X_test)

# Make prediction
y_pred = engine.predict_classes(X_test)
proba_pred = engine.predict_proba(X_test)

print("Prediction result:")
for i in range(len(proba_pred)):
        print("    instance id: {}, class: {}, proba: {}".format(i+1, y_pred[i], proba_pred[i]))
