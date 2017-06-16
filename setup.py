import os, sys
import numpy as np
from distutils.core import setup, Extension

MKLROOT = os.environ.get('MKLROOT', '/opt/intel/mkl')

# Find MKL library despite slight differences between MacOS and Linux installation paths.
for libext in ['lib', 'lib/intel64']:
    rt = os.path.join(MKLROOT, libext, 'libmkl_rt.so')
    if os.path.exists( rt ):
        print("Using MKL lib at %s" % rt)
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
	'-L' + os.path.join(MKLROOT, 'lib', 'intel64'), '-lmkl_rt',
    ]
)


setup(name='lowrank',
      version='0.95',
      description='Locally Low-Rank Regularization Routines',
      author='Michael Driscoll',
      author_email='driscoll@cs.berkeley.edu',
      url='https://github.com/mbdriscoll/lowrank',
      ext_modules = [lowrank],
     )
