#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include "lowrank.h"

static PyObject*
svthresh_py(PyObject *self, PyObject *args)
{
    float thresh;
    PyArrayObject *py_imgs;
    int b, sx, sy, sz; // block size, shift x, y, z.
    if (!PyArg_ParseTuple(args, "Ofiiii", &py_imgs, &thresh, &b, &sx, &sy, &sz))
        return NULL;

    int T0=1, P=1, T1=1, X=1, Y=1, Z=1;
    switch (PyArray_NDIM(py_imgs)) {
        default:
        case 6: T0 = PyArray_DIM(py_imgs, 5);
        case 5: P  = PyArray_DIM(py_imgs, 4);
        case 4: T1 = PyArray_DIM(py_imgs, 3);
        case 3: X  = PyArray_DIM(py_imgs, 2);
        case 2: Y  = PyArray_DIM(py_imgs, 1);
        case 1: Z  = PyArray_DIM(py_imgs, 0);
    }

    complex float *imgs = PyArray_DATA(py_imgs);

    svthresh( thresh, b, sx, sy, sz, T0, P, T1, X, Y, Z, imgs );

    Py_RETURN_NONE;
}


static PyMethodDef lowrankMethods[] = {
    { "svthresh", svthresh_py, METH_VARARGS, NULL },
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
