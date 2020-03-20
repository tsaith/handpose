
import pytest
import numpy as np
from numpy.testing import assert_allclose
from handpose.tracking import *
from handpose.sensor_fusion import *
from handpose.utils import Quaternion

def test_ref_projector():

    vec0 = np.array([1, 0, 0])

    # Construct a reference projector
    projector = RefProjector()

    # Roate to the reference axes
    theta = 0.0 / 180 * np.pi
    phi = 30.0 / 180 *np.pi
    q_ref = Quaternion.from_spherical(theta, phi)

    # Vector in the reference axes
    vec_ref = q_ref.rotate_vector(vec0)
    projector.q_ref = q_ref # Save the reference quaternion

    # Rotate to the target axes
    theta = 0.0 / 180 * np.pi
    phi = -60.0 / 180 *np.pi
    q_target = Quaternion.from_spherical(theta, phi)

    # Vector representation in the target axes
    vec_target = q_target.rotate_vector(vec0)

    # Projected vector
    vec_proj = projector.project_vector(vec_target)
    vec_gt = np.array([0.0, -1.0, 0.0])

    assert_allclose(vec_proj, vec_gt, atol=1e-5)

if __name__ == '__main__':
    pytest.main([__file__])
