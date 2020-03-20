from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(ext_modules = cythonize(Extension(
           "fast_madgwick", # the extesion name
           sources=["fast_madgwick.pyx", "madgwick_ahrs.cpp"],  # the source files
           extra_compile_args=["-std=c++11"],
           language="c++",                       # generate and compile C++ code
           include_dirs=[np.get_include()], # include path
           library_dirs=[], # lib path
           libraries=[],
      )))
