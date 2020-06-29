from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extensions = [
    Extension("processing_module", sources=["motion.pyx"], language="c++")
]
setup(ext_modules = cythonize(extensions))
