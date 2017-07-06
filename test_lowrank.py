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
             [10,11],
             [11,12],
             [12,13],
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


def sv_thresh_py(t, X):
    # Performs batch-K svd threshold on M x N x K array using c-ordering
    assert X.ndim == 3
    assert X.flags['C']
    for i, x in enumerate(X):
        U, s, V = np.linalg.svd(x, full_matrices=False)
        s = np.maximum(s - t, 0)
        S = np.diag(s)
        x[:] = np.dot(U, np.dot(S,V))

@pytest.mark.parametrize("T0,W,T1,b,lamda",
    product( [1,2,8,9],
             [1,2,8,9],
             [1,2,8,9],
             [3,8,9],
             [1.0,0.5,0.0] ))
def test_svthresh6_rich(T0, W, T1, b, lamda):
    act = rand64c(T0,W,T1,b,b,b)

    exp = act.copy()                         # T0, W, T1, b, b, b
    exp = np.moveaxis(exp, [0,2], [-2,-1])   # W, b, b, b, T0, T1
    exp = exp.reshape((W, b*b*b, T0*T1))     # W, bbb, T
    exp = np.ascontiguousarray(exp)
    sv_thresh_py(lamda, exp)
    exp = exp.reshape((W, b, b, b, T0, T1))  # W, b, b, b, T0, T1
    exp = np.moveaxis(exp, [-2, -1], [0, 2]) # T0, W, T1, b, b, b
    
    lowrank.svthresh6(act, lamda)

    np.testing.assert_allclose(act, exp, rtol=1e-2)


@pytest.mark.parametrize("T,Xdb,b,lamda",
    product( [1,2,8,9],
             [1,2,8,9],
             [3,8,9],
             [1.0,0.5,0.0] ))
def test_svthresh4_rich(T, Xdb, b, lamda):
    X = Y = Z = Xdb * b
    act = rand64c(T, X, Y, Z)

    exp = act.copy()                                   # T, X, Y, Z
    exp = exp.reshape( (T, X//b, b, Y//b, b, Z//b, b)) # T, x, b, y, b, z, b
    exp = np.transpose(exp, [1,3,5,2,4,6,0])           # x, y, z, b, b, b, T
    exp = exp.reshape( (-1, b**3, T) )                 # W, bbb, T
    exp = np.ascontiguousarray(exp)
    sv_thresh_py(lamda, exp)
    exp = exp.reshape( (X//b, Y//b, Z//b, b, b, b, T) ) # x, y, z, b, b, b, T
    exp = np.transpose(exp, [6,0,3,1,4,2,5])            # T, x, b, y, b, z, b
    exp = exp.reshape((T,X,Y,Z))
    
    lowrank.svthresh4(act, lamda, b, 0, 0, 0)

    np.testing.assert_allclose(act, exp, rtol=1e-2)


def benchmark_svthresh4():
    b = 8
    lamda = 0.5
    T, X, Y, Z = 20, 208, 308, 480
    ntrials = 5

    imgs = rand64c(T,X,Y,Z)
    sx, sy, sz = np.random.randint(0, b, 3)
    nblocks = np.prod(imgs.shape) / b**3  # approximate
    nflops = nblocks * 2 * 5 * (b*b*b*T*min(b*b*b,T))

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
