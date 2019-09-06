import pytest
import numpy as np
from numpy.testing import assert_allclose
from handpose.preprocessing import *


def test_scale_standard():

    X = np.array([[ 1., -1.,  2.],
                  [ 2.,  0.,  0.],
                  [ 0.,  1., -1.]])   

    ground_true = np.array([[ 0.,        -1.22474487,  1.33630621],
                            [ 1.22474487, 0.        , -0.26726124],
                            [-1.22474487, 1.22474487, -1.06904497]])

    X_scaled, scaler = scale(X, method='standard') 


    assert_allclose(X_scaled, ground_true)

def test_scale_minmax():

    X = np.array([[ 1., -1.,  2.],
                  [ 2.,  0.,  0.],
                  [ 0.,  1., -1.]])   

    ground_true = np.array([[ 0.5,         0.,          1.        ],
                            [ 1.,          0.5,         0.33333333],
                            [ 0.,          1.,          0.        ]])

    X_scaled, scaler = scale(X, method='minmax') 


    assert_allclose(X_scaled, ground_true)


if __name__ == '__main__':
    pytest.main([__file__])
