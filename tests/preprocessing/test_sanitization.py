import pytest
import numpy as np
from numpy.testing import assert_allclose
from handpose.preprocessing import *


def test_detect_na():
    

    a = np.ones((3, 3))
    a[1,1] = np.NaN

    has_na = detect_na(a)

    assert has_na == True


def test_fill_na():
    

    ground_true = np.zeros((3, 3))

    a = ground_true.copy()
    a[1,1] = np.NaN

    filled = fill_na(a, value=0)

    assert_allclose(filled, ground_true)


if __name__ == '__main__':
    pytest.main([__file__])
