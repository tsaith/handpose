import pytest
import numpy as np
from numpy.testing import assert_allclose
from handpose.tracking import *


def test_ego_tracker():

    # Define frames with gray-scale
    shape = (32, 32, 1)

    prev = np.zeros(shape, dtype=np.uint8)
    curr = np.zeros(shape, dtype=np.uint8)

    # Bounding box
    bbox = (8, 8, 16, 16)

    # Define tracker
    tracker = EgoTracker(prev, bbox)

    # Update tracker
    tracker.update(curr)
    is_motional = tracker.is_motional()

    assert_allclose(is_motional, False)


if __name__ == '__main__':
    pytest.main([__file__])
