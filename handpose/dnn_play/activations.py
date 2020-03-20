import numpy as np

def sigmoid(x):
    """
    Return the sigmoid (aka logistic) function, 1 / (1 + exp(-x)).
    """
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    """
    Return the first derivative of the sigmoid function.
    """
    f = sigmoid(x)
    df = f*(1.0-f)
    return df

def tanh(x):
    """
    Return the tanh(x).
    """
    return np.tanh(x)

def tanh_deriv(x):
    """
    Return the first derivative of tanh(x).
    """
    return 1.0 - np.tanh(x)**2

def relu(x):
    """
    Return the rectified linear unit = max(0, x).
    """
    return np.maximum(x, 0)

def relu_deriv(x):
    """
    Return the first derivative of rectified linear unit.
    """
    return np.where(x > 0, 1.0, 0.0)


from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'activation function')
