import os
import numpy as np
from distutils.core import setup, Extension

MKLROOT = '/opt/intel/mkl'

lowrank = Extension('lowrank',
    sources=['lowrank.c'],
    include_dirs=[
        np.get_include(),
        os.path.join(MKLROOT, 'include'),
    ],
    extra_compile_args = ['-std=c11', '-fopenmp', '-m64', '-O3'],
    extra_link_args=['-fopenmp', '-mavx',
        #'-L' + os.path.join(MKLROOT, 'lib'),
        os.path.join(MKLROOT, 'lib', 'libmkl_intel_lp64.a'),
        os.path.join(MKLROOT, 'lib', 'libmkl_sequential.a'),
        os.path.join(MKLROOT, 'lib', 'libmkl_core.a'),
        '-lgomp', '-lpthread', '-lm', '-ldl',
    ]
)


setup(name='lowrank',
      version='0.95',
      description='Locally Low-Rank Regularization Routines',
      author='Michael Driscoll',
      author_email='driscoll@cs.berkeley.edu',
      url='https://github.com/mbdriscoll/lowrank',
      packages=['lowrank'],
      ext_modules = [lowrank],
     )
