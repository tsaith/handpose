import numpy as np
from handpose.vib import has_gesture
from handpose.vib import VibConfig, VibModel
from handpose.utils import set_cuda_visible_devices

# Device configuration
set_cuda_visible_devices("") # CPU only

# Model path
dir_path ="../data/vib/models"

model_file = 'vib_model.hdf5'

model_path = "{}/{}".format(dir_path, model_file)

# Initialize the gesture engine
config = VibConfig()
config.model_path = model_path
model = VibModel(config)


# Testing
num_tdata = 700
num_dims = 3

X_test = np.ones((num_tdata, num_dims))

# Preprocess the features
X_test = model.preprocess(X_test)

# Make prediction
proba_pred = model.predict_proba(X_test)
y_pred = np.argmax(proba_pred)


print("Prediction result:")
print("    class: {}, proba: {}".format(y_pred, proba_pred))
