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

    complex float *imgs = PyArray_DATA(py_imgs);
    int Z = PyArray_DIM(py_imgs, 0),
        Y = PyArray_DIM(py_imgs, 1),
        X = PyArray_DIM(py_imgs, 2),
        T = PyArray_DIM(py_imgs, 3);

    svthresh( thresh, b, sx, sy, sz, T, X, Y, Z, imgs );

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
