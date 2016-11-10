#!/usr/bin/env python
# python setup_dtw.py build_ext --inplace
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

sourcefiles = ['dtw.pyx', 'cdtw.c']

extensions = [Extension("dtw", sourcefiles)]

setup(
    ext_modules = cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
