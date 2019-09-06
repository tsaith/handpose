import cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
cimport fast_madgwick

from handpose.utils import Quaternion

# Define type
ctypedef np.float64_t FLOAT64_t

cdef class FastMadgwick:
    cdef Madgwick* target

    def __cinit__(self, float sample_period, float beta):

        self.target = new Madgwick(sample_period, beta)

    def __dealloc__(self):
        del self.target

    def update_ahrs(self, FLOAT64_t [:] gyro, FLOAT64_t [:] accel, FLOAT64_t [:] mag):

        self.target.update(gyro[0], gyro[1], gyro[2],
                           accel[0], accel[1], accel[2],
                           mag[0], mag[1], mag[2])

    def update_imu(self, FLOAT64_t [:] gyro, FLOAT64_t [:] accel):

        self.target.updateIMU(gyro[0], gyro[1], gyro[2],
                              accel[0], accel[1], accel[2])

    def get_roll(self):
        return self.target.getRollRadians()

    def get_pitch(self):
        return self.target.getPitchRadians()

    def get_yaw(self):
        return self.target.getYawRadians()

    def get_quat_array(self):

        cdef vector[float] cpp_array = self.target.get_quat_array()

        array = np.zeros(4, dtype=np.float64)
        array[0] = cpp_array[0]
        array[1] = cpp_array[1]
        array[2] = cpp_array[2]
        array[3] = cpp_array[3]

        return array


    @property
    def quat(self): # q_se, sensor axes respective to the Earth axes
        q = Quaternion.from_array(self.get_quat_array())
        return q

    def get_counter(self):
        return self.target.get_counter()
