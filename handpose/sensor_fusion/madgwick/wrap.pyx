import cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
cimport wrap

ctypedef np.float64_t DTYPE_t

cdef class FastMadgwick:
    cdef Madgwick* target

    def __cinit__(self, float sample_period, float beta, np.ndarray[DTYPE_t, ndim=1] q):

        self.target = new Madgwick(sample_period, beta, q[0], q[1], q[2], q[3])

    def __dealloc__(self):
        del self.target

    def update_ahrs(self,
        np.ndarray[DTYPE_t, ndim=1] gyro,
        np.ndarray[DTYPE_t, ndim=1] accel,
        np.ndarray[DTYPE_t, ndim=1] mag):

        self.target.update(gyro[0], gyro[1], gyro[2],
                           accel[0], accel[1], accel[2],
                           mag[0], mag[1], mag[2])

    def update_imu(self,
        np.ndarray[DTYPE_t, ndim=1] gyro,
        np.ndarray[DTYPE_t, ndim=1] accel):

        self.target.updateIMU(gyro[0], gyro[1], gyro[2],
                              accel[0], accel[1], accel[2])

    def get_roll(self):
        return self.target.getRollRadians()

    def get_pitch(self):
        return self.target.getPitchRadians()

    def get_yaw(self):
        return self.target.getYawRadians()
