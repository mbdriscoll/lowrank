#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdlib.h>
#include <complex.h>

#define MKL_Complex8  complex float
#define MKL_Complex16 complex double

#include <mkl.h>
#include <mkl_types.h>

#define min(x,y) (((x)<(y))?(x):(y))
#define max(x,y) (((x)>(y))?(x):(y))

// macro for indexing into a block
#define IDX_BLK(t,bx,by,bz) \
   (( t) * (b*b*b) + \
    (bx) * (  b*b) + \
    (by) * (    b) + \
    (bz) * (    1))

// macro for indexing into an img
#define IDX_IMG(t,x,y,z) \
  ((t) * (X*Y*Z) + \
   (x) * (  Y*Z) + \
   (y) * (    Z) + \
   (z) * (    1))


static PyObject*
svthresh(PyObject *self, PyObject *args)
{
    float thresh;
    PyArrayObject *py_imgs;
    int b, sx, sy, sz; // block size, shift x, y, z.
    if (!PyArg_ParseTuple(args, "Ofiiii", &py_imgs, &thresh, &b, &sx, &sy, &sz))
        return NULL;

    complex float *imgs = PyArray_DATA(py_imgs);
    int Z = PyArray_DIM(py_imgs, 0),
        Y = PyArray_DIM(py_imgs, 1),
        X = PyArray_DIM(py_imgs, 2),
        T = PyArray_DIM(py_imgs, 3);

    int M = b*b*b,
        N = T;
    int K = min(M,N);
    int ldc = M,
        ldu = M,
        ldv = K;

    #pragma omp parallel
    {
        const complex float alpha = 1.0, beta = 0.0;
        complex float *U     = malloc( M * K * sizeof(complex float) );
        complex float *V     = malloc( K * N * sizeof(complex float) );
        float         *s     = malloc(     K * sizeof(        float) );
        float    *superb     = malloc(     K * sizeof(        float) );
        complex float *block = malloc( M * N * sizeof(complex float) );

        // for every block...
        #pragma omp for collapse(3)
        for (int x = -sx; x < X; x += b) {
        for (int y = -sy; y < Y; y += b) {
        for (int z = -sz; z < Z; z += b) {

            // block[:] = 0
            memset(block, 0, M * N * sizeof(complex float));

            // load block
            for (int t  = 0; t  < T;  t++) {
            for (int bx = 0; bx < b; bx++) { if (0 <= x+bx && x+bx < X) {
            for (int by = 0; by < b; by++) { if (0 <= y+by && y+by < Y) {
            for (int bz = 0; bz < b; bz++) { if (0 <= z+bz && z+bz < Z) {
                  block[ IDX_BLK(t,bx,by,bz) ] = imgs[ IDX_IMG(t,x+bx,y+by,z+bz) ];
            }}}}}}}

            // U, s, V = svd(block)
            LAPACKE_cgesvd( LAPACK_COL_MAJOR, 'S', 'S',
                M, N, block, ldc, s, U, ldu, V, ldv, superb );

            // sV = thresh(s) * V
            for (int k = 0; k < K; k++)
                for (int n = 0; n < N; n++)
                    V[k*N+n] *= max(s[k]-thresh, 0);

            // block = U * sV
            cblas_cgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                 M, N, K, &alpha, U, ldu, V, ldv, &beta, block, ldc );

            // restore block
            for (int t  = 0; t  < T;  t++) {
            for (int bx = 0; bx < b; bx++) { if (0 <= x+bx && x+bx < X) {
            for (int by = 0; by < b; by++) { if (0 <= y+by && y+by < Y) {
            for (int bz = 0; bz < b; bz++) { if (0 <= z+bz && z+bz < Z) {
                  imgs[ IDX_IMG(t,x+bx,y+by,z+bz) ] = block[ IDX_BLK(t,bx,by,bz) ];
            }}}}}}}

        }}}

        free(block); free(U); free(V); free(s); free(superb);
    }

    Py_RETURN_NONE;
}
static PyMethodDef lowrankMethods[] = {
    { "svthresh", svthresh, METH_VARARGS, NULL },
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef lowrankmodule = {
    PyModuleDef_HEAD_INIT, "lowrank", NULL, -1, lowrankMethods,
};

PyMODINIT_FUNC
PyInit_lowrank(void)
{
    import_array();
    return PyModule_Create(&lowrankmodule);
}
