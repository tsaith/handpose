from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(ext_modules = cythonize(Extension(
           "wrap", # the extesion name
           sources=["wrap.pyx", "madgwick_ahrs.cpp"],  # the source files
           language="c++",                       # generate and compile C++ code
           include_dirs=[np.get_include()], # include path
           library_dirs=[], # lib path
           libraries=[],
      )))
