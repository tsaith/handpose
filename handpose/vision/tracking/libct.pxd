from libcpp.vector cimport vector
from ..libcore cimport CvRect, CvRect, CvMat

ctypedef unsigned char uint8

cdef extern from "ct.h":
    cdef cppclass CompressiveTracker:
        CompressiveTracker() except +
        void init_wrap(vector[vector[uint8]] &_frame, vector[int] &_objectBox)
        void process_frame_wrap(vector[vector[uint8]] &_frame, vector[int] &_objectBox)

    void c_get_feature_values_v1(
        vector[vector[double]] &feature_values,
        vector[vector[int]] boxes,
        vector[vector[vector[int]]] features,
        vector[vector[double]] weights,
        vector[vector[double]] integral_image)

    void c_get_feature_values(
        vector[vector[double]] &feature_values,
        vector[CvRect] boxes,
        vector[vector[CvRect]] features,
        vector[vector[double]] weights,
        vector[vector[double]] integral_image)

    void c_ratio_classifier(
        int &index_ratio_max, double &ratio_max,
        vector[double] mu_pos, vector[double] sigma_pos,
        vector[double] mu_neg, vector[double] sigma_neg,
        vector[vector[double]] feature_values)


cdef class CyCompressiveTracker:
    cdef CompressiveTracker *this_ptr
