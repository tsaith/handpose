import numpy as np
import quaternion
#from numba import jit, njit, int32, float32, float64

class Quaternion(quaternion.quaternion):
    """
    A simple class implementing basic quaternion arithmetic.
    """
    def __init__(self, w=None, x=None, y=None, z=None):
        """
        Initializes a Quaternion object
        :param w_or_q: A scalar representing the real part of the quaternion, another Quaternion object or a
                    four-element array containing the quaternion values
        :param x: The first imaginary part if w_or_q is a scalar
        :param y: The second imaginary part if w_or_q is a scalar
        :param z: The third imaginary part if w_or_q is a scalar
        """
        super().__init__(w, x, y, z)
        #self._quat = np.quaternion(w, x, y, z)

    def to_angle_axis(self):
        """
        Returns the quaternion's rotation represented by an Euler angle and axis.
        If the quaternion is the identity quaternion (1, 0, 0, 0), a rotation along the x axis with angle 0 is returned.
        :return: rad, x, y, z
        """

        q = self.to_array()
        q0 = q[0]
        rad = 2.0*np.arccos(q0)

        sin_part = np.sqrt(1.0-q0*q0)
        if abs(sin_part) < 1e-8:
            return np.array([1.0, 0, 0, 0])

        x = q[1] / sin_part
        y = q[2] / sin_part
        z = q[3] / sin_part

        return rad, x, y, z


    def rotate_vector(self, v):
        return quaternion.rotate_vectors(self, v)

    def rotate_axes(self, v):
        return quaternion.rotate_vectors(self.inverse(), v)

    def inv(self):
        q_inv = self.inverse()
        w = q_inv.w
        x = q_inv.x
        y = q_inv.y
        z = q_inv.z
        return Quaternion(w, x, y, z)

    def to_array(self):
        return quaternion.as_float_array(self)

    def to_spherical(self):
        """
        Convert the quaternion into the spherical coordinates.
        Returns
        -------
        coords: array
            Array of spherical coordinates; [theta, phi], where
            theta is the poloidal angle and phi is the toroidal angle.
            The unit z vector is the reference vector.
        """
        return quaternion.as_spherical_coords(self)

    @staticmethod
    def from_angle_axis(rad, x, y, z):
        s = np.sin(0.5*rad)
        return Quaternion(np.cos(0.5*rad), x*s, y*s, z*s)

    @staticmethod
    def from_array(a):
        return Quaternion(a[0], a[1], a[2], a[3])

    @staticmethod
    def from_spherical(theta, phi):
        """
        Return the quaternion from the spherical coordinates.

        Parameters
        ----------
        theta: float
            Poloidal angle.
        phi: float
            Toroidal angle.

        Returns
        -------
        quat: quaternion
             Rotational quaternion.
        """
        q = quaternion.from_spherical_coords(theta, phi)
        a = quaternion.as_float_array(q)
        return Quaternion(a[0], a[1], a[2], a[3])

    @property
    def roll(self):
        # Roll angle in radians

        q0 = self.w
        q1 = self.x
        q2 = self.y
        q3 = self.z
        roll = np.arctan2(q0*q1 + q2*q3, 0.5 - q1*q1 - q2*q2)

        return roll

    @property
    def pitch(self):
        # Pitch angle in radians

        q0 = self.w
        q1 = self.x
        q2 = self.y
        q3 = self.z
        pitch = np.arcsin(-2.0 * (q1*q3 - q0*q2))

        return pitch

    @property
    def yaw(self):
        # Yaw angle in radians

        q0 = self.w
        q1 = self.x
        q2 = self.y
        q3 = self.z
        yaw = np.arctan2(q1*q2 + q0*q3, 0.5 - q2*q2 - q3*q3)

        return yaw

