import pytest
from handpose.metrics import average_accuracy
import numpy as np
from numpy.testing import assert_allclose



def test_average_accuracy():

    y_true = np.array([1.0, 1.0, 1.0])
    y_pred = np.array([1.1, 1.1, 1.1])
    acc = average_accuracy(y_true, y_pred, full_range=1.0)
    acc_true = 0.98999999999999999
     

    assert np.abs(acc - acc_true) < 1e-8 


if __name__ == '__main__':
    pytest.main([__file__])
