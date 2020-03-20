import pytest
import numpy as np
from numpy.testing import assert_allclose

from dnn_play.utils.np_utils import to_binary_class_matrix, flatten_struct, pack_struct


def init_weights(layer_units):

    n_layers = len(layer_units)
    weights = [{} for i in range(n_layers - 1)]

    for i in range(n_layers - 1):

        weights[i]['W'] = np.random.randn(layer_units[i], layer_units[i+1])
        weights[i]['b'] = np.random.randn(layer_units[i+1])

    return weights

def test_flatten_struct():

    layer_units = (4, 8, 4)
    n_layers = len(layer_units)
    weights = init_weights(layer_units)

    actual  = flatten_struct(weights)
    desired = []
    for i in range(n_layers - 1):
        desired.append(weights[i]['W'].flatten())
        desired.append(weights[i]['b'].flatten())
    desired = np.concatenate(desired)

    assert_allclose(actual, desired, atol=1e-8)

def test_pack_struct():
    layer_units = (4, 8, 4)
    n_layers = len(layer_units)

    num = 0
    for i in range(n_layers - 1):
        num += layer_units[i+1]*layer_units[i] + layer_units[i+1]
    weights = np.random.randn(num)

    actual  = flatten_struct(pack_struct(weights, layer_units))
    desired = weights

    assert_allclose(actual, desired, atol=1e-8)
