import numpy as np
cimport numpy as np
cimport libcore

cdef class Point2i:
    def __cinit__(self, int x, int y):
        self.this_ptr = new CvPoint2i()
        self.this_ptr.x = x
        self.this_ptr.y = y
    def __dealloc__(self):
        del self.this_ptr
    property x:
        def __get__(self): return self.this_ptr.x
        def __set__(self, val): self.this_ptr.x = val
    property y:
        def __get__(self): return self.this_ptr.y
        def __set__(self, val): self.this_ptr.y = val

cdef class Point:
    def __cinit__(self, int x, int y):
        self.this_ptr = new CvPoint()
        self.this_ptr.x = x
        self.this_ptr.y = y
    def __dealloc__(self):
        del self.this_ptr
    property x:
        def __get__(self): return self.this_ptr.x
        def __set__(self, val): self.this_ptr.x = val
    property y:
        def __get__(self): return self.this_ptr.y
        def __set__(self, val): self.this_ptr.y = val

cdef class Rect:
    def __cinit__(self, int x, int y, int width, int height):
        self.this_ptr = new CvRect()
        self.this_ptr.x = x
        self.this_ptr.y = y
        self.this_ptr.width  = width
        self.this_ptr.height = height
    def __dealloc__(self):
        del self.this_ptr
    property x:
        def __get__(self): return self.this_ptr.x
        def __set__(self, val): self.this_ptr.x = val
    property y:
        def __get__(self): return self.this_ptr.y
        def __set__(self, val): self.this_ptr.y = val
    property width:
        def __get__(self): return self.this_ptr.width
        def __set__(self, val): self.this_ptr.width = val
    property height:
        def __get__(self): return self.this_ptr.height
        def __set__(self, val): self.this_ptr.height = val
