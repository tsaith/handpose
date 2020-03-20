from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(ext_modules = cythonize(Extension(
           "vision/lib/libcore",                            # the extesion name
           sources=["vision/libcore.pyx", "vision/lib/types.cpp"],  # the source files
           language="c++",                       # generate and compile C++ code
           include_dirs=[np.get_include(), "vision/include", "/usr/local/opt/opencv3/include"],
      )))

setup(ext_modules = cythonize(Extension(
           "vision/lib/libct",                            # the extesion name
           sources=["vision/tracking/libct.pyx", "vision/tracking/compressive_tracker.cpp", "vision/tracking/ct_engine.cpp"],  # the source files
           language="c++",                       # generate and compile C++ code
           include_dirs=[np.get_include(), "vision/include", "/usr/local/opt/opencv3/include"],
           library_dirs=["/usr/local/opt/opencv3/lib"],
           libraries=["opencv_core", "opencv_imgproc", "opencv_highgui", "opencv_videoio"],
      )))

