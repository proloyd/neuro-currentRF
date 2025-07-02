# Author: Proloy Das <email:proloyd94@gmail.com>
# License: BSD (3-clause) 
from setuptools import setup, find_packages, Extension
import numpy as np

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = False

# Cython extensions
ext = '.pyx' if cythonize else '.c'
ext_cpp = '.pyx' if cythonize else '.cpp'
extensions = [
    Extension('ncrf.opt', [f'ncrf/opt{ext}']),
    Extension('ncrf.dsyevh3C.dsyevh3py', [f'ncrf/dsyevh3C/dsyevh3py{ext_cpp}'], include_dirs=['ncrf/dsyevh3C']),
]
if cythonize:
    extensions = cythonize(extensions)


setup(
    include_dirs=[np.get_include()],
    packages=find_packages(),
    ext_modules=extensions,
)
