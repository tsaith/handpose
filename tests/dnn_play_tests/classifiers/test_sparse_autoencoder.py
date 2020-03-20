import pytest
import numpy as np
from numpy.testing import assert_allclose

from dnn_play.classifiers.sparse_autoencoder import SparseAutoencoder, sparse_autoencoder_loss, rel_err_gradients

def test_gradients():
    """
    Test if the gradients is correct.
    """
    rel_err = rel_err_gradients()
    assert rel_err < 1e-8

if __name__ == '__main__':
    pytest.main([__file__])
