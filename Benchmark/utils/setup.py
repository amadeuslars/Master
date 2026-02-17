from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extension = Extension(
    "operators2",
    sources=["operators2.pyx"],
    include_dirs=[numpy.get_include()],
)

setup(
    name="operators2",
    ext_modules=cythonize([extension])
)