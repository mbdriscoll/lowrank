import os, sys
import numpy as np
from distutils.core import setup, Extension

MKLROOT = os.environ.get('MKLROOT', '/opt/intel/mkl')

# Find MKL library despite slight differences between MacOS and Linux installation paths.
for libext in ['lib', 'lib/intel64']:
    core = os.path.join(MKLROOT, libext, 'libmkl_core.a')
    seq  = os.path.join(MKLROOT, libext, 'libmkl_sequential.a')
    lp  = os.path.join(MKLROOT, libext, 'libmkl_intel_lp64.a')
    if os.path.exists( core ):
        print("Using MKL lib at %s" % core)
        break
else:
    print("Could not locate MKL lib dir.")
    sys.exit(1)


lowrank = Extension('lowrank',
    sources=['pylowrank.c', 'lowrank.c'],
    include_dirs=[
        np.get_include(),
        os.path.join(MKLROOT, 'include'),
    ],
    extra_compile_args = ['-std=c11', '-fopenmp', '-m64', '-O3', '-DMKL_ILP64'],
    extra_link_args=['-fopenmp', '-mavx',
        lp, seq, core,
        '-lpthread', '-lm', '-ldl',
    ]
)


setup(name='lowrank',
      version='0.95',
      description='Locally Low-Rank Regularization Routines',
      author='Michael Driscoll',
      author_email='driscoll@cs.berkeley.edu',
      url='https://github.com/mbdriscoll/lowrank',
      packages=['cuda'],
      ext_modules = [lowrank],
     )
