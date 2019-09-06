import numpy as np

from handpose.vib import  VibEnsemble, VibConfig, VibModel
from handpose.utils import get_reduced_proba

# Please note that this example is incomplete for execution.

# Model path
dir_path ="."
model_file = 'vib_model.hdf5'
model_path = "{}/{}".format(dir_path, model_file)

# Initialize the gesture engine
config = VibConfig()
config.model_path = model_path
model = VibModel(config)


# Define the ensemble model
sampling_rate = 700
num_preds = 2 # Number of predictions
vib_en = VibEnsemble(sampling_rate=sampling_rate, num_preds=num_preds, model=model)

# Prepare the simulated acceleration with shape of (num_tdata, DOF)
# accel_simu = ...

# Simulate the process of prediction
n_samples = 3

nt = len(accel_simu)
for it in range(nt):
    pd = vib_en.update(accel_simu[it, :], threshold=0.15)

    if pd is not None:
        reduced_p = get_reduced_proba(pd)
        print("it = {}".format(it))
        print("pd = {}".format(pd))
        print("reduced_p = ", reduced_p)

