import cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

cdef class CyCompressiveTracker:
    def __cinit__(self, frame, object_box):
        self.this_ptr = new CompressiveTracker()

        cdef vector[int] c_object_box
        c_object_box.push_back(object_box.x)
        c_object_box.push_back(object_box.y)
        c_object_box.push_back(object_box.width)
        c_object_box.push_back(object_box.height)

        self.this_ptr.init_wrap(frame, c_object_box)

    def __dealloc__(self):
        del self.this_ptr

    def process_frame(self, frame, object_box):

        cdef vector[int] c_object_box
        c_object_box.push_back(object_box.x)
        c_object_box.push_back(object_box.y)
        c_object_box.push_back(object_box.width)
        c_object_box.push_back(object_box.height)

        self.this_ptr.process_frame_wrap(frame, c_object_box)

        object_box.x = c_object_box[0]
        object_box.y = c_object_box[1]
        object_box.width  = c_object_box[2]
        object_box.height = c_object_box[3]

        return object_box

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_get_feature_values(
    np.ndarray[object, ndim=1, mode="c"] boxes,
    np.ndarray[object, ndim=2, mode="c"] features,
    np.ndarray[np.float64_t, ndim=2, mode="c"] weights,
    np.ndarray[np.float64_t, ndim=2, mode="c"] integral_image):

    cdef int n_boxes = boxes.size
    cdef int n_features = features.shape[0]
    feature_values = np.zeros((n_features, n_boxes), dtype=np.float64)

    cdef int n_rects
    cdef int i, j, k

    cdef int x_min, x_max
    cdef int y_min, y_max
    cdef double weight, value

    cdef int box_x, box_y
    cdef int rect_x, rect_y
    cdef int rect_width, rect_height

    # Get the feature values
    for i in range(n_features):
        n_rects = 4
        for j in range(n_boxes):
            value = 0.0
            for k in range(n_rects):

                box_x = boxes[j].x
                box_y = boxes[j].y

                rect_x = features[i, k].x
                rect_y = features[i, k].y
                rect_width  = features[i, k].width
                rect_height = features[i, k].height

                x_min = box_x + rect_x
                y_min = box_y + rect_y
                x_max = x_min + rect_width
                y_max = y_min + rect_height

                weight = weights[i, k]
                value += weight * (integral_image[y_min, x_min] +
                                   integral_image[y_max, x_max] -
                                   integral_image[y_min, x_max] -
                                   integral_image[y_max, x_min])
            feature_values[i, j] = value

    return feature_values

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_ratio_classifier(
    np.ndarray[np.float64_t, ndim=1, mode="c"] mu_pos,
    np.ndarray[np.float64_t, ndim=1, mode="c"] sigma_pos,
    np.ndarray[np.float64_t, ndim=1, mode="c"] mu_neg,
    np.ndarray[np.float64_t, ndim=1, mode="c"] sigma_neg,
    np.ndarray[np.float64_t, ndim=2, mode="c"] feature_values):
    """
    Ratio classifier.

    Return
    ------
    index_ratio_max : integer.
        The index of maximal ratio.

    ratio_max : float.
        The value of maximal ratio.
    """
    cdef int index_ratio_max = 0
    cdef double ratio_max = -1.0e30

    c_ratio_classifier(
        index_ratio_max, ratio_max,
        mu_pos, sigma_pos, mu_neg, sigma_neg,
        feature_values)

    return index_ratio_max, ratio_max
