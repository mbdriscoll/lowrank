import pytest
import numpy as np
from itertools import product

import lowrank

def rand64c(*shape):
    arr = np.random.rand(*shape) + 1j*np.random.rand(*shape)
    arr = np.require(arr, dtype=np.complex64, requirements='F')
    return arr

@pytest.mark.parametrize("T,X,Y,Z,b,lamda",
    product( [1,2,8,9],
             [120,141],
             [121,142],
             [122,143],
             [3,8,9],
             [1.0,0.5,0.0] ))
def test_svthresh(T, X, Y, Z, b, lamda):
    """ weak test: just call the C routine and see if that works """
    imgs = rand64c(Z,Y,X,T)
    sx, sy, sz = np.random.randint(0, b, 3)
    lowrank.svthresh(imgs, lamda, b, sx, sy, sz)
