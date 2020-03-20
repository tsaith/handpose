import numpy as np
from handpose.vib import has_gesture
from handpose.engine import EngineConfig, VibEngine
from handpose.utils import set_cuda_visible_devices

# Device configuration
set_cuda_visible_devices("") # CPU only

# Model path
dir_path ="../data/vib/models"

model_file = 'model.ckpt-4'
graph_file = model_file + ".meta"

model_path = "{}/{}".format(dir_path, model_file)
graph_path = "{}/{}".format(dir_path, graph_file)

# Initialize the gesture engine
config = EngineConfig()
config.model_path = model_path
config.graph_path = graph_path
engine = VibEngine(config)


# Testing
fs = 700
num_dims = 3
num_features = fs*num_dims

X_test = np.ones(num_features)

# Detect if there is a gesture
gesture_existed = has_gesture(X_test)

# Preprocess the features
X_test = np.expand_dims(X_test, axis=0)
X_test = engine.preprocess(X_test)

# Make prediction
y_pred = engine.predict_classes(X_test)
proba_pred = engine.predict_proba(X_test)

print("Has a gesture? {}".format(gesture_existed))

print("Predictions:")
for i in range(len(proba_pred)):
        print("    instance id: {}, class: {}, proba: {}".format(i+1, y_pred[i], proba_pred[i]))
