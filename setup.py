from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler.Options import directive_defaults
import os
import numpy as np

os.environ['CC'] = 'gcc-4.9'
os.environ['CXX'] = 'g++-4.9'

directive_defaults['linetrace'] = True
directive_defaults['binding'] = True

ext = [ 
    Extension(
      name='hmm', 
      sources=['hmm.pyx'],
      extra_compile_args=['-O3', '-Wno-unused-function', '-ffast-math'],
      define_macros=[('CYTHON_TRACE', '1')],
      include_dirs=[np.get_include()],
    )
    ]

setup(ext_modules=cythonize(ext))
