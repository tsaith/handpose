import pytest
from dnn_play.activations import sigmoid, tanh, relu
import numpy as np
from numpy.testing import assert_allclose

from scipy.special import expit


def get_test_values():
    return np.array([-1.0, 0.0, 1.0])

def test_sigmoid():
    x = get_test_values()
    actual  = sigmoid(x)
    desired = expit(x)

    assert_allclose(actual, desired, rtol=1e-5)

def test_tanh():
    x = get_test_values()
    actual  = tanh(x)
    desired = np.tanh(x)

    assert_allclose(actual, desired, rtol=1e-5)

def test_relu():
    """
    Note relu(x) = max(0, x).
    """

    x = np.array([-1.0, 0.0, 1.0])
    actual = relu(x)
    desired = np.array([0.0, 0.0, 1.0])

    assert_allclose(actual, desired, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
