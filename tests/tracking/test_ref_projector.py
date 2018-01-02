import pytest
import numpy as np
from numpy.testing import assert_allclose
from handpose.tracking import RefProjector
from handpose.utils import Quaternion


def test_ref_projector():

    # Unit vector
    vec0 = np.array([1, 0, 0])

    # Reference projector
    projector = RefProjector()

    theta = 0.0 / 180 * np.pi
    phi = 30.0 / 180 *np.pi
    q_ref = Quaternion.from_spherical(theta, phi)
    # Set the reference quaternion
    projector.q_ref = q_ref

    theta = 0.0 / 180 * np.pi
    phi = -60.0 / 180 *np.pi
    q_target = Quaternion.from_spherical(theta, phi)

    vec_target = q_target.rotate_vector(vec0)
    vec_proj = projector.project_vector(vec_target)
    vec_proj_gt = np.array([0.0, -1.0, 0.0])

    assert_allclose(vec_proj, vec_proj_gt, atol=1e-5)

if __name__ == '__main__':
    pytest.main([__file__])
