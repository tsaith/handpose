import pytest
import numpy as np
from numpy.testing import assert_allclose
from handpose.utils import fourier_spectrum


def test_fourier_spectrum():

    ground_true = np.array([1.262177e-31, 1.000000e+00, 4.000000e+00,
                            6.812569e-32, 1.466206e-31])

    fs = 10.0  # Sampling rate
    dt = 1.0/fs # Sampling interval
    t = np.arange(0, 1, dt) # Time array

    f1 = 1 # Frequency 1 (Hz)
    f2 = 2 # Frequency 2 (Hz)
    y = 2*np.cos(2*np.pi*f1*t) + 4*np.cos(2*np.pi*f2*t)

    spectrum, _ = fourier_spectrum(y, dt, spectrum_type='power')

    assert_allclose(spectrum, ground_true, rtol=1e-03)


if __name__ == '__main__':
    pytest.main([__file__])
