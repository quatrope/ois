#include <Python.h>
//#include <numpy/npy_common.h>
//#include <numpy/ndarrayobject.h>
#include <numpy/arrayobject.h>
#include "oistools.h"

#if PY_MAJOR_VERSION >= 3
#define PY3
#endif

static PyObject *
varconv_gen_matrix_system(PyObject *self, PyObject *args)
{
    PyObject *py_sciimage, *py_refimage;
    PyArrayObject *np_mask;
    int k_side;
    int kernel_polydeg; // The degree of the varying polynomial for the kernel
    int bkg_deg; // The degree of the varying polynomial for the background
    unsigned char hasmask;

    if (!PyArg_ParseTuple(args, "OObOiii", 
            &py_sciimage, &py_refimage, &hasmask, &np_mask,
            &k_side, &kernel_polydeg, &bkg_deg)) {
        return NULL;
    }
    PyArrayObject *np_sciimage = (PyArrayObject *)PyArray_FROM_OTF(py_sciimage, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (np_sciimage == NULL) {
        Py_XDECREF(np_sciimage);
        return NULL;
    }
    PyArrayObject *np_refimage = (PyArrayObject *)PyArray_FROM_OTF(py_refimage, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (np_refimage == NULL) {
        Py_XDECREF(np_sciimage);
        Py_XDECREF(np_refimage);
        return NULL;
    }
    if (NULL == np_mask) return NULL;

    double* sciimage = (double*)PyArray_DATA(np_sciimage);
    double* refimage = (double*)PyArray_DATA(np_refimage);
    int n = (int)PyArray_DIM(np_sciimage, 0);
    int m = (int)PyArray_DIM(np_sciimage, 1);

    char* mask;
    if (hasmask == 1) {
        mask = (char*)np_mask->data;
    } else {
        mask = NULL;
    }

    lin_system result_sys = build_matrix_system(\
    n, m, sciimage, refimage, k_side, k_side,\
    kernel_polydeg, bkg_deg, mask);

    Py_DECREF(np_sciimage);
    Py_DECREF(np_refimage);

    int total_dof = result_sys.b_dim;
    npy_intp Mdims[2] = {total_dof, total_dof};
    npy_intp bdims = total_dof;
    PyArrayObject* pyM = (PyArrayObject *)PyArray_SimpleNewFromData(2, Mdims, NPY_DOUBLE, result_sys.M);
    PyArray_ENABLEFLAGS(pyM, NPY_ARRAY_OWNDATA);
    PyArrayObject* pyb = (PyArrayObject *)PyArray_SimpleNewFromData(1, &bdims, NPY_DOUBLE, result_sys.b);
    PyArray_ENABLEFLAGS(pyb, NPY_ARRAY_OWNDATA);

    return Py_BuildValue("NN", pyM, pyb);
}


static PyObject *
varconv_convolve2d_adaptive(PyObject *self, PyObject *args) {
    PyArrayObject *np_image, *np_kernelcoeffs;
    int k_polydeg; // The degree of the varying polynomial

    if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &np_image,
            &PyArray_Type, &np_kernelcoeffs, &k_polydeg))  return NULL;
    if (NULL == np_image) return NULL;
    if (NULL == np_kernelcoeffs) return NULL;

    int n = np_image->dimensions[0];
    int m = np_image->dimensions[1];
    int k_height = np_kernelcoeffs->dimensions[0];
    int k_width = np_kernelcoeffs->dimensions[1];

    double* image = (double*)np_image->data;
    double* k_coeffs = (double*)np_kernelcoeffs->data;
    
    double* Conv = (double*)calloc(n * m, sizeof(*Conv));
    convolve2d_adaptive(n, m, image, k_height, k_width, k_polydeg, k_coeffs, Conv);

    npy_intp Convdims[2] = {n, m};
    PyObject* pyConv = PyArray_SimpleNewFromData(2, Convdims, NPY_DOUBLE, Conv);

    return Py_BuildValue("O", pyConv);

}

static PyMethodDef VarConvMethods[] = {
    {"gen_matrix_system", varconv_gen_matrix_system, METH_VARARGS,
     "Generate the matrix system to find best convolution parameters."},
    {"convolve2d_adaptive", varconv_convolve2d_adaptive, METH_VARARGS,
     "Convolves image with an adaptive kernel."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


#ifdef PY3
static struct PyModuleDef varconvmodule = {
   PyModuleDef_HEAD_INIT,
   "varconv",   /* name of module */
   NULL, /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   VarConvMethods
};

PyMODINIT_FUNC
PyInit_varconv(void)
{
    PyObject *m;
    m = PyModule_Create(&varconvmodule);
    import_array();
    return m;
}
#else
PyMODINIT_FUNC
initvarconv(void)
{
    (void) Py_InitModule("varconv", VarConvMethods);
    import_array();
}
#endif
