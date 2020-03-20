import numpy as np
import h5py


def fix_keras_model_file(model_file):
    """
    Fix bug in keras model file.
    """
    with h5py.File(model_file, 'a') as f:
        if 'optimizer_weights' in f.keys():
            del f['optimizer_weights']
        f.close()
