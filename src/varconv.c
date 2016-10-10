#include <Python.h>
//#include <numpy/npy_common.h>
#include <numpy/ndarrayobject.h>
#include <numpy/arrayobject.h>

static PyObject *
varconv_cconvolve_var(PyObject *self, PyObject *args)
{
    PyArrayObject *vecin, *vecout;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &vecin,
            &PyArray_Type, &vecout))  return NULL;
    if (NULL == vecin)  return NULL;
    if (NULL == vecout)  return NULL;

    int n = vecin->dimensions[0];
    double* cin = (double*)vecin->data;

    printf("The last index is %g\n", cin[n-1]);
    cin[n-1] = 33.0;

    return Py_BuildValue("i", 42);
}

static PyMethodDef VarConvMethods[] = {
    {"cconvolve_var", varconv_cconvolve_var, METH_VARARGS,
     "Do a 2D convolution with a modulated kernel"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initvarconv(void)
{
    (void) Py_InitModule("varconv", VarConvMethods);
    import_array();
}
