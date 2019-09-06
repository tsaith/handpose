import pytest
import handpose as hp
from handpose.preprocessing import window_method
import numpy as np
from numpy.testing import assert_allclose



def test_window_method():
    
    num_rows = 3
    num_cols = 2
    num_elements = num_rows * num_cols
    
    a = np.array([i for i in range(num_elements)]).reshape((num_rows, num_cols))
    b = window_method(a, win_size=1)
    b_true = np.array([[0, 1, 2, 3], [2, 3, 4, 5]])  

    assert_allclose(b, b_true)



if __name__ == '__main__':
    pytest.main([__file__])
