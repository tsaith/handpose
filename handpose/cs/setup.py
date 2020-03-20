from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(ext_modules = cythonize(Extension(
           "wrap", # the extesion name
           sources=["wrap.pyx", "compressive_tracker.cpp"],  # the source files
           extra_compile_args=["-std=c++11"],
           language="c++",                       # generate and compile C++ code
           include_dirs=[np.get_include(), "/home/andrew/anaconda2/envs/py3/include/opencv2"], # please check the OpenCv include path
           library_dirs=["/home/andrew/anaconda2/envs/py3/lib"], # plase check the OpenCv lib path
           libraries=["opencv_core", "opencv_shape", "opencv_video","opencv_xobjdetect" ,
                      "opencv_rgbd", "opencv_imgproc", "opencv_highgui", "opencv_videoio"],
      )))

"""
setup(ext_modules = cythonize(Extension(
           "wrap", # the extesion name
           sources=["wrap.pyx", "compressive_tracker.cpp"],  # the source files
           extra_compile_args=["-std=c++11"],
           language="c++",                       # generate and compile C++ code
           include_dirs=[np.get_include(), "/usr/local/opt/opencv3/include"], # please check the OpenCv include path
           library_dirs=["/usr/local/opt/opencv3/lib"], # plase check the OpenCv lib path
           libraries=["opencv_core", "opencv_imgproc", "opencv_highgui", "opencv_videoio"],
      )))
"""


