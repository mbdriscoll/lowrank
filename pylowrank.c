#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <assert.h>

#include <Python.h>
#include <numpy/arrayobject.h>

#include "lowrank.h"

#if 0

Case 1: local LLR
imgs.shape = (T, X, Y, Z)
svthresh(imgs, sx, sy, sz)

Case 2: dist LLR
imgs.shape = (T0, B, T1, BX, BY, BZ)

T0: # processors
B : # blocks per processor
T1: # timepoints per processor
BX: # block size
svthresh(imgs, 0, 0, 0)

#endif

static PyObject*
svthresh4_py(PyObject *self, PyObject *args)
{
    float thresh;
    PyArrayObject *py_imgs;
    int b, sx, sy, sz; // block size, shift x, y, z.
    if (!PyArg_ParseTuple(args, "Ofiiii", &py_imgs, &thresh, &b, &sx, &sy, &sz))
        return NULL;

    assert( PyArray_NDIM(py_imgs) == 4 );

    complex float *imgs = PyArray_DATA(py_imgs);
    int T0 = 1;
    int  B = 1;
    int T1 = PyArray_DIM(py_imgs, 0), // timepoints per proc
         X = PyArray_DIM(py_imgs, 1), // block x
         Y = PyArray_DIM(py_imgs, 2), // block y
         Z = PyArray_DIM(py_imgs, 3); // blocy z

    svthresh( thresh, imgs, T0, B, T1, X, Y, Z, b, sx, sy, sz );

    Py_RETURN_NONE;
}

static PyObject*
svthresh6_py(PyObject *self, PyObject *args)
{
    float thresh;
    PyArrayObject *py_imgs;
    if (!PyArg_ParseTuple(args, "Of", &py_imgs, &thresh))
        return NULL;

    assert( PyArray_NDIM(py_imgs) == 6 );

    complex float *imgs = PyArray_DATA(py_imgs);
    int T0 = PyArray_DIM(py_imgs, 0), // outer time dim
         K = PyArray_DIM(py_imgs, 1), // images dim
        T1 = PyArray_DIM(py_imgs, 2), // inner time dim
         X = PyArray_DIM(py_imgs, 3), // block x
         Y = PyArray_DIM(py_imgs, 4), // block y
         Z = PyArray_DIM(py_imgs, 5); // blocy z

    int B = X;
    assert( B == X );
    assert( B == Y );
    assert( B == Z );

    svthresh( thresh, imgs, T0, K, T1, X, Y, Z, B, 0, 0, 0 );

    Py_RETURN_NONE;
}


static PyMethodDef lowrankMethods[] = {
    { "svthresh4", svthresh4_py, METH_VARARGS, NULL },
    { "svthresh6", svthresh6_py, METH_VARARGS, NULL },
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
