
from libcpp.vector cimport vector

ctypedef unsigned char uint8

cdef extern from "madgwick_ahrs.h":
    cdef cppclass Madgwick:
        Madgwick() except +
        Madgwick(float samplePeriod, float beta) except +
        void begin(float sampleFrequency)
        void update(float gx, float gy, float gz, float ax, float ay, float az, float mx, float my, float mz)
        void updateIMU(float gx, float gy, float gz, float ax, float ay, float az)
        float getRoll()
        float getPitch()
        float getYaw()
        float getRollRadians()
        float getPitchRadians()
        float getYawRadians()
        float get_q0()
        float get_q1()
        float get_q2()
        float get_q3()
        vector[float] get_quat_array()
        int get_counter()

