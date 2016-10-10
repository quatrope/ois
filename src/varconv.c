#include <Python.h>
//#include <numpy/npy_common.h>
#include <numpy/ndarrayobject.h>
#include <numpy/arrayobject.h>

static PyObject *
varconv_cconvolve_var(PyObject *self, PyObject *args)
{
    PyArrayObject *np_image, *np_kernel;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &np_image,
            &PyArray_Type, &np_kernel))  return NULL;
    if (NULL == np_image)  return NULL;
    if (NULL == np_kernel)  return NULL;

    int n = np_image->dimensions[0];
    int m = np_image->dimensions[1];
    int kn = np_kernel->dimensions[0];
    int km = np_kernel->dimensions[1];
    int w = kn / 2;

    double* image = (double*)np_image->data;
    double* kernel = (double*)np_kernel->data;

    double* CRKn = (double*)malloc(n * m * sizeof(*CRKn));

    //convolution
    for (int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            for (int mm = 0; mm < km; mm++){
                for(int nn = 0; nn < kn; nn++){
                    int ii = i + (mm - w); //index of convolution//
                    int jj = j + (nn - w); //index of convolution//
                    if (ii >= 0 && ii < n && jj >= 0 && jj < n) {
                        CRKn[i + j * n] += image[ii + jj * n] * kernel[mm + nn * kn];
                    }
                }
            }
        }
    }

    npy_intp dims[2] = {n, m};
    PyObject* cconv = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, CRKn);

    //return Py_BuildValue("i", 42);
    return cconv;
}


PyArrayObject *pymatrix(PyObject *objin)  {
    return (PyArrayObject *) PyArray_ContiguousFromObject(objin,
        NPY_DOUBLE, 2,2);
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
