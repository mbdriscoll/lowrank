import time
import pytest
import numpy as np
from itertools import product

import lowrank

def rand64c(*shape):
    arr = np.random.rand(*shape) + 1j*np.random.rand(*shape)
    arr = np.require(arr, dtype=np.complex64, requirements='C')
    return arr

@pytest.mark.parametrize("T,X,Y,Z,b,lamda",
    product( [1,2,8,9],
             [120,141],
             [121,142],
             [122,143],
             [3,8,9],
             [1.0,0.5,0.0] ))
def test_svthresh4(T, X, Y, Z, b, lamda):
    """ weak test: just call the C routine and see if that works """
    imgs = rand64c(T,X,Y,Z)
    sx, sy, sz = np.random.randint(0, b, 3)
    lowrank.svthresh4(imgs, lamda, b, sx, sy, sz)


@pytest.mark.parametrize("T0,W,T1,b,lamda",
    product( [1,2,8,9],
             [1,2,8,9],
             [1,2,8,9],
             [3,8,9],
             [1.0,0.5,0.0] ))
def test_svthresh6(T0, W, T1, b, lamda):
    """ weak test: just call the C routine and see if that works """
    imgs = rand64c(T0,W,T1,b,b,b)
    lowrank.svthresh6(imgs, lamda)


def benchmark_svthresh4():
    b = 8
    lamda = 0.5
    T, X, Y, Z = 20, 208, 308, 480
    ntrials = 5

    imgs = rand64c(T,X,Y,Z)
    sx, sy, sz = np.random.randint(0, b, 3)

    times = []
    for trial in range(ntrials):
        start = time.time()
        lowrank.svthresh4(imgs, lamda, b, sx, sy, sz)
        times.append( time.time() - start )
    
    sec = np.median(times)
    nblocks = X*Y*Z // b**3  # approximate
    nflops = nblocks * 5 * (T*T*b**3)
    gflops_sec = nflops / sec * 1e-9

    print("(local layout) median of %d trials: %2.2f seconds, %4.2f GFlops/sec" % (ntrials, sec, gflops_sec))


def benchmark_svthresh6():
    b = 8
    lamda = 0.5
    T0, W, T1 = 4, 208*308*480//(b*b*b), 5
    ntrials = 5

    imgs = rand64c(T0,W,T1,b,b,b)

    times = []
    for trial in range(ntrials):
        start = time.time()
        lowrank.svthresh6(imgs, lamda)
        times.append( time.time() - start )
    
    sec = np.median(times)
    nblocks = W
    T = T0 * T1
    nflops = nblocks * 5 * (T*T*b**3)
    gflops_sec = nflops / sec * 1e-9

    print("(dist layout)  median of %d trials: %2.2f seconds, %4.2f GFlops/sec" % (ntrials, sec, gflops_sec))

if __name__ == '__main__':
    benchmark_svthresh4()
    benchmark_svthresh6()
