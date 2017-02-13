import pytest
import numpy as np
from numpy.testing import assert_allclose
from handpose.utils import maximum_multi


def test_maximum_multi():

    n = 3
    a = np.array([0.1, -0.2, 0.03])
    b = np.array([0.01, 0.2, -0.3])
    c = np.array([-0.1, 0.02, 0.3])

    out = maximum_multi([a, b, c])
    ground_true = np.array([0.1, 0.2, 0.3])

    assert_allclose(out, ground_true)


if __name__ == '__main__':
    pytest.main([__file__])
