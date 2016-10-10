#include <Python.h>

static PyObject *
varconv_cconvolve_var(PyObject *self, PyObject *args)
{
    /*const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);*/
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
}
