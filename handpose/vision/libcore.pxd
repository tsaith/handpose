cdef extern from "opencv2/core/core.hpp" namespace "cv":
    cdef cppclass CvPoint2i "cv::Point2i":
        Point2i() except +
        Point2i(int x, inty) except +
        int x
        int y

    cdef cppclass CvPoint "cv::Point":
        Point() except +
        Point(double x, double y) except +
        double x
        double y

    cdef cppclass CvRect "cv::Rect":
        Rect() except +
        Rect(int x, int y, int width, int height) except +
        int x
        int y
        int width
        int height

    cdef cppclass CvMat "cv::Mat":
        Mat() except +
        Mat(int rows, int cols, int type) except +
        void create(int rows, int cols, int type)
        int rows
        int cols
        int dims
        void* data

cdef class Point2i:
    cdef CvPoint2i *this_ptr

cdef class Point:
    cdef CvPoint *this_ptr

cdef class Rect:
    cdef CvRect *this_ptr

cdef class Mat:
    cdef CvMat *this_ptr
