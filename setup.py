# Author: Proloy Das <email:proloyd94@gmail.com>
# License: BSD (3-clause) 
from setuptools import setup, find_packages
from glob import glob
from distutils.extension import Extension
# from Cython.Distutils import build_ext
from os.path import pathsep
try:
    import numpy as np
except ModuleNotFoundError:
    import pkg_resources
    print([p.project_name for p in pkg_resources.working_set])

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
