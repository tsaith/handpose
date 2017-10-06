
from libcpp.vector cimport vector

ctypedef unsigned char uint8

cdef extern from "madgwick_ahrs.h":
    cdef cppclass Madgwick:
        Madgwick() except +
        Madgwick(float samplePeriod, float beta, float q0, float q1, float q2, float q3) except +
        void begin(float sampleFrequency)
        void update(float gx, float gy, float gz, float ax, float ay, float az, float mx, float my, float mz)
        void updateIMU(float gx, float gy, float gz, float ax, float ay, float az)
        float getRoll()
        float getPitch()
        float getYaw()
        float getRollRadians()
        float getPitchRadians()
        float getYawRadians()

