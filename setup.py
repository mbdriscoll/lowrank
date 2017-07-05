import os, sys
import numpy as np
from distutils.core import setup, Extension

MKLROOT = os.environ.get('MKLROOT', '/opt/intel/mkl')

lowrank = Extension('lowrank',
    sources=['pylowrank.c', 'lowrank.c'],
    include_dirs=[
        np.get_include(),
        os.path.join(MKLROOT, 'include'),
    ],
    extra_compile_args = ['-std=c11', '-fopenmp', '-m64', '-O3', '-DMKL_ILP64'],
    extra_link_args=['-fopenmp', '-mavx',
        '-L' + os.path.join(MKLROOT, 'lib', 'intel64'), '-lmkl_rt',
        '-lpthread', '-lm', '-ldl']
)


setup(name='lowrank',
      version='0.95',
      description='Locally Low-Rank Regularization Routines',
      author='Michael Driscoll',
      author_email='driscoll@cs.berkeley.edu',
      url='https://github.com/mbdriscoll/lowrank',
      ext_modules = [lowrank],
     )
