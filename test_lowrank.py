import time
import pytest
import numpy as np
from itertools import product

import lowrank

def rand64c(*shape):
    arr = np.random.rand(*shape) + 1j*np.random.rand(*shape)
    arr = np.require(arr, dtype=np.complex64, requirements='F')
    return arr

@pytest.mark.parametrize("T0,P,T1,X,Y,Z,b,lamda",
    product( [1,2,3],
             [1,2,3],
             [1,2,8,9],
             [10,11],
             [11,12],
             [12,13],
             [3,8,9],
             [1.0,0.5,0.0] ))
def test_svthresh_6d(T0, P, T1, X, Y, Z, b, lamda):
    """ weak test: just call the C routine and see if that works """
    imgs = rand64c(Z,Y,X,T1,P,T0)
    sx, sy, sz = np.random.randint(0, b, 3)
    lowrank.svthresh(imgs, lamda, b, sx, sy, sz)

@pytest.mark.parametrize("T,X,Y,Z,b,lamda",
    product( [1,2,8,9],
             [10,11],
             [11,12],
             [12,13],
             [3,8,9],
             [1.0,0.5,0.0] ))
def test_svthresh_4d(T, X, Y, Z, b, lamda):
    """ weak test: just call the C routine and see if that works """
    imgs = rand64c(Z,Y,X,T)
    sx, sy, sz = np.random.randint(0, b, 3)
    lowrank.svthresh(imgs, lamda, b, sx, sy, sz)

def benchmark_svthresh():
    b = 8
    lamda = 0.5
    T, X, Y, Z = 20, 208, 308, 480
    T, X, Y, Z = 2, 28, 38, 48
    ntrials = 5

    imgs = rand64c(Z,Y,X,T)
    sx, sy, sz = np.random.randint(0, b, 3)

    times = []
    for trial in range(ntrials):
        start = time.time()
        lowrank.svthresh(imgs, lamda, b, sx, sy, sz)
        times.append( time.time() - start )
    
    sec = np.median(times)
    nblocks = np.prod(imgs.shape) / b**3  # approximate
    nflops = nblocks * 5 * (T*T*b**3)
    gflops_sec = nflops / sec * 1e-9

    print("Median of %d trials: %2.2f seconds, %4.2f GFlops/sec" % (ntrials, sec, gflops_sec))


if __name__ == '__main__':
    benchmark_svthresh()
