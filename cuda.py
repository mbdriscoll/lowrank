import numpy as np
from ctypes import *
from numba import cuda
from itertools import product

from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba.cuda.cudadrv.drvapi import cu_device_ptr, cu_stream
from numba.cuda.cudadrv import driver

gomp = CDLL('libgomp.so.1', mode=RTLD_GLOBAL)

# ---------------------------------------------------------------------------
# common
# ---------------------------------------------------------------------------
class device_ptr(c_ulong):
    @staticmethod
    def from_param(obj):
        if isinstance(obj, DeviceNDArray):
            return obj.device_ctypes_pointer
        else:
            raise TypeError("expected DeviceNDArray, got %s" % type(obj))


# ---------------------------------------------------------------------------
# cuSolver
# ---------------------------------------------------------------------------
cusolver = CDLL("libcusolver.so")

class cusolverDnHandle_t(c_void_p):
    def __init__(self):
        status = cusolverDnCreate( byref(self) )
        assert status == cusolverStatus_t.SUCCESS, status

    def __del__(self):
        status = cusolverDnDestroy(self)
        assert status == cusolverStatus_t.SUCCESS, status

    def buffersize(self, m, n):
        lwork = c_int()
        status = cusolverDnCgesvd_bufferSize( self, m, n, byref(lwork) )
        assert status == cusolverStatus_t.SUCCESS, status
        return lwork.value

    def setstream(self, stream):
        status = cusolverDnSetStream(self, stream.handle)
        assert status == cusolverStatus_t.SUCCESS, status

    def cgesvd(self, U, S, VT, A, work, lwork, devinfo, stream=None):
        if stream:
            self.setstream(stream)
        else:
            assert False, stream
        m, n = U.shape[0], VT.shape[0]
        jobu = jobv = c_char(b'A')
        lda, ldu, ldvt = m, m, n
        status = cusolverDnCgesvd(self, jobu, jobv, m, n, A, lda, S, U, ldu, VT, ldvt,
            work, lwork, None, devinfo)
        assert status == cusolverStatus_t.SUCCESS, status


class cusolverStatus_t(c_int):
    SUCCESS = 0
    NOT_INITIALIZED = 1
    ALLOC_FAILED = 2
    INVALID_VALUE = 3
    ARCH_MISMATCH = 4
    MAPPING_ERROR = 5
    EXECUTION_FAILED = 6
    INTERNAL_ERROR = 7
    MATRIX_TYPE_NOT_SUPPORTED = 8
    NOT_SUPPORTED = 9
    ZERO_PIVOT = 10
    INVALID_LICENSE = 11


cusolverDnCreate = cusolver.cusolverDnCreate
cusolverDnCreate.rettype = cusolverStatus_t
cusolverDnCreate.argtypes = ( POINTER(cusolverDnHandle_t), )

cusolverDnDestroy = cusolver.cusolverDnDestroy
cusolverDnDestroy.rettype = cusolverStatus_t
cusolverDnDestroy.argtypes = ( cusolverDnHandle_t, )

cusolverDnCgesvd_bufferSize = cusolver.cusolverDnCgesvd_bufferSize
cusolverDnCgesvd_bufferSize.rettype = cusolverStatus_t
cusolverDnCgesvd_bufferSize.argtypes = (
    cusolverDnHandle_t,
    c_int, c_int, POINTER(c_int),
)

cusolverDnSetStream = cusolver.cusolverDnSetStream
cusolverDnSetStream.rettype = cusolverStatus_t
cusolverDnSetStream.argtypes = (cusolverDnHandle_t, cu_stream)

cusolverDnCgesvd = cusolver.cusolverDnCgesvd
cusolverDnCgesvd.rettype = cusolverStatus_t
cusolverDnCgesvd.argtypes = (
    cusolverDnHandle_t, # handle
    c_char, c_char,     # jobu jobv
    c_int, c_int,       # m, n
    device_ptr, c_int,  # A, lda
    device_ptr,         # S
    device_ptr, c_int,  # U, ldu
    device_ptr, c_int,  # VT, ldvt
    device_ptr, c_int,  # work, lwork
    c_void_p,           # rwork
    device_ptr,         # devInfo
)


# ---------------------------------------------------------------------------
# cuBLAS
# ---------------------------------------------------------------------------
cublas = CDLL("libcublas.so")

class cublasHandle_t(c_void_p):
    def __init__(self):
        status = cublasCreate( byref(self) )
        assert status == cublasStatus_t.SUCCESS, status

    def __del__(self):
        status = cublasDestroy( self )
        assert status == cublasStatus_t.SUCCESS, status

    def setstream(self, stream):
        status = cublasSetStream(self, stream.handle)
        assert status == cublasStatus_t.SUCCESS, status

    def cgemm(self, C, A, B, stream=None):
        if stream:
            self.setstream(stream)
        (m, k), n = A.shape, B.shape[1]
        lda = A.shape[0]
        ldb = B.shape[0]
        ldc = A.shape[0]
        alpha = (c_float * 2)(1,0)
        beta  = (c_float * 2)(0,0)
        status = cublasCgemm(self, cublasOperation_t.N, cublasOperation_t.C,
            m, n, k, alpha, A, lda, B, ldb, beta, C, ldc )
        assert status == cublasStatus_t.SUCCESS, status

    def cdgmm(self, C, x, A, stream=None):
        if stream:
            self.setstream(stream)
        m, n = A.shape
        lda = ldc = A.shape[0]
        left = cublasSideMode_t.LEFT
        status = cublasCdgmm(self, left, m, n, A, lda, x, 1, C, ldc)
        assert status == cublasStatus_t.SUCCESS, status


class cublasOperation_t(c_uint):
    N = 0
    T = 1
    C = 2


class cublasStatus_t(c_int):
    SUCCESS         = 0
    NOT_INITIALIZED = 1
    ALLOC_FAILED    = 3
    INVALID_VALUE   = 7
    ARCH_MISMATCH   = 8
    MAPPING_ERROR   = 11
    EXECUTION_FAILED= 13
    INTERNAL_ERROR  = 14
    NOT_SUPPORTED   = 15
    LICENSE_ERROR   = 16


cublasCgemm = cublas.cublasCgemm_v2
cublasCgemm.rettype = cublasStatus_t
cublasCgemm.argtypes = (
    cublasHandle_t,      # handle
    cublasOperation_t,   # transa
    cublasOperation_t,   # transb
    c_int, c_int, c_int, # m, n, k,
    POINTER(c_float*2),  # alpha
    device_ptr, c_int,   # A, lda
    device_ptr, c_int,   # B, ldb
    POINTER(c_float*2),  # beta
    device_ptr, c_int,   # C, ldc
)

class cublasSideMode_t(c_int):
    LEFT  = 0
    RIGHT = 1


cublasCdgmm = cublas.cublasCdgmm
cublasCdgmm.rettype = cublasStatus_t
cublasCdgmm.argtypes = (
    cublasHandle_t, cublasSideMode_t,
    c_int, c_int,      # m, n
    device_ptr, c_int, # A, lda
    device_ptr, c_int, # x, incx
    device_ptr, c_int, # C, ldc
)

cublasCreate = cublas.cublasCreate_v2
cublasCreate.rettype = cublasStatus_t
cublasCreate.argtypes = ( POINTER(cublasHandle_t), )

cublasDestroy = cublas.cublasDestroy_v2
cublasDestroy.rettype = cublasStatus_t
cublasDestroy.argtypes = ( cublasHandle_t, )

cublasSetStream = cublas.cublasSetStream_v2
cublasSetStream.rettype = cublasStatus_t
cublasSetStream.argtypes =  (cublasHandle_t, cu_stream)


# ---------------------------------------------------------------------------
# custom kernels
# ---------------------------------------------------------------------------

@cuda.jit
def img2blk(img, blk, x, y, z):
    tid = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    if tid >= blk.size: return
    T, X, Y, Z = img.shape
    t  = tid // (X*Y*Z)
    bx = tid // (  Y*Z)
    by = tid // (    Z)
    bz = tid  % (    Z)
    if (0<=(x+bx)<X and 0<=(y+by)<Y and 0<=(z+bz)<Z):
        blk[t,bx,by,bz] = img[t,x+bx,y+by,z+bz]

    
@cuda.jit
def blk2img(img, blk, x, y, z):
    tid = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    if tid >= blk.size: return
    T, X, Y, Z = img.shape
    t  = tid // (X*Y*Z)
    bx = tid // (  Y*Z)
    by = tid // (    Z)
    bz = tid  % (    Z)
    if (0<=(x+bx)<X and 0<=(y+by)<Y and 0<=(z+bz)<Z):
        img[t,x+bx,y+by,z+bz] = blk[t,bx,by,bz]

@cuda.jit
def thresh(sc, sf, t):
    tid = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    if tid >= sf.size: return
    sc[tid] = max(sf[tid]-t, 0)


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------

class Worker(object):
    def __init__(self, solver, blas, imgs_shape, b):
        self.solver = solver 
        self.blas = blas
        T, X, Y, Z = imgs_shape
        m, n = b*b*b, T
        assert m >= n, 'cusolver limitation: %s >= %s' % (m, n)
        k = min(m,n)
        A_shape = (T, b, b, b)
        self.stream = s = cuda.stream()
        self.lwork  = lwork = solver.buffersize(m,n)
        self.A      = cuda.device_array(A_shape, dtype=np.complex64, stream=s)
        self.U      = cuda.device_array(  (m,m), dtype=np.complex64, stream=s)
        self.Sf     = cuda.device_array(      k, dtype=np.float32,   stream=s)
        self.Sc     = cuda.device_array(      k, dtype=np.complex64, stream=s)
        self.VT     = cuda.device_array(  (n,n), dtype=np.complex64, stream=s)
        self.work   = cuda.device_array(  lwork, dtype=np.complex64, stream=s)
        self.devinfo= cuda.device_array(      1, dtype=np.int32,     stream=s)

    def svthresh1(self, imgs, lamda, x, y, z ):
        A, U, Sf, Sc, VT = self.A, self.U, self.Sf, self.Sc, self.VT
        stream = s = self.stream
        tpb = 128
        
        # zero out block
        nbytes = A.size * A.dtype.itemsize
        driver.device_memset(A, 0, nbytes, stream=s)

        # gather casorati block
        blocks = (A.size + tpb - 1) // tpb
        img2blk[ blocks, tpb, s ]( imgs, A, x, y, z )

        # compute svd
        self.solver.cgesvd(U, Sf, VT, A, self.work, self.lwork, self.devinfo, stream=s)

        # threshold
        blocks = (Sf.size + tpb - 1) // tpb
        thresh[ blocks, tpb, s ]( Sc, Sf, lamda )

        # rebuild block
        self.blas.cdgmm(VT, Sc, VT, stream=s)
        self.blas.cgemm(A, U, VT, stream=s)

        # scatter block
        blocks = (A.size + tpb - 1) // tpb
        blk2img[ blocks, tpb, s ]( imgs, A, x, y, z )
        

def svthresh(imgs, lamda, b, sx, sy, sz):
    import sys
    nstreams = int(sys.argv[1])

    T, X, Y, Z = imgs.shape

    solver = cusolverDnHandle_t()
    blas = cublasHandle_t()

    workers = [ Worker(solver, blas, imgs.shape, b) for w in range(nstreams) ]

    for i, (x, y, z) in enumerate(product(
        range(-sx, X, b),
        range(-sy, Y, b),
        range(-sz, Z, b),
    )):
        w = workers[ i % nstreams ]
        w.svthresh1( imgs, lamda, x, y, z )

    cuda.synchronize()


def rand64c(*shape):
    arr = np.random.rand(*shape) + 1j*np.random.rand(*shape)
    arr = np.require(arr, dtype=np.complex64)
    return arr

if __name__ == '__main__':
    T, X, Y, Z = 2, 1, 3, 12, 13, 14
    lamda = 2.0
    b = 8
    s = (0,0,0)

    imgs = cuda.to_device( rand64c(T,X,Y,Z) )

    svthresh(imgs, lamda, b, *s)
