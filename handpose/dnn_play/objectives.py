import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_pred - y_true))

from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'objective function')
