from setuptools import setup, find_packages
from glob import glob
from distutils.extension import Extension
# from Cython.Distutils import build_ext
from os.path import pathsep
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
    name="ncrf",
    description="A Unified Approach to MEG Source Analysis under the Continuous Stimuli Paradigm",
    long_description='add-on module to eelbrain for neural current RF estimation'
                     'GitHub: https://github.com/proloyd/neuro-currentRF',
    version="0.4dev",
    python_requires='>=3.6',

    install_requires=[
        'eelbrain',
    ],

    # metadata for upload to PyPI
    author="Proloy DAS",
    author_email="proloy@umd.com",
    license="BSD 2-Clause (Simplified)",
    # cmdclass={'build_ext': build_ext},
    include_dirs=[np.get_include()],
    packages=find_packages(),
    ext_modules=extensions,
    url='https://github.com/proloyd/neuro-currentRF',
    project_urls={
        "Source Code": "https://github.com/proloyd/neuro-currentRF/archive/0.3.tar.gz",
    }
)
