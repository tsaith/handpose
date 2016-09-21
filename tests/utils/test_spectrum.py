import pytest
import numpy as np
from numpy.testing import assert_allclose
from handpose.utils import fourier_spectrum


def test_fourier_spectrum():

    ground_true = np.array([3.15544362e-32, 1.00000000e+00, 4.00000000e+00,
                            3.27979796e-32, 1.24622240e-30])

    fs = 10.0  # Sampling rate
    dt = 1.0/fs # Sampling interval
    t = np.arange(0, 1, dt) # Time array

    f1 = 1 # Frequency 1 (Hz)
    f2 = 2 # Frequency 2 (Hz)
    y = 2*np.cos(2*np.pi*f1*t) + 4*np.cos(2*np.pi*f2*t)

    spectrum, _ = fourier_spectrum(y, dt, spectrum_type='power')

    assert_allclose(spectrum, ground_true)


if __name__ == '__main__':
    pytest.main([__file__])
