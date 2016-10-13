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

    double* Conv = (double*)malloc(n * m * sizeof(*CRKn));

    kernel = calloc(kernel_height * kernel_width, sizeof(*kernel));
    Conv = calloc(n * m, sizeof(*Conv));

    for (int conv_row = 0; conv_row < n; ++conv_row) {
        for (int conv_col = 0; conv_col < m; ++conv_col) {
            conv_index = conv_row * m + conv_col

            for (int p = 0; p < kernel_height; ++p) {
                for (int q = 0; q < kernel_width; ++q) {
                    kpq = p * kn + q;
                    if (kpq > 0) kernel[kpq - 1] = 0.0;
                    kernel[kpq] = 1.0;

                    for (int exp_x = 0; exp_x < deg_x; ++exp_x) {
                        p_pow = pow(p, exp_y);
                        for (int exp_y = 0; exp_y < deg_y; ++exp_y) {
                            q_pow = pow(q, exp_x);
                            img_index = conv_index - p * m - q;
                            // make sure img_index is in bounds of image
                            Conv[conv_index] += image[img_index] * p_pow * q_pow
                        } // exp_y
                    } // exp_x

                } //q
            } // p

        } // conv_col
    } // conv_row



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
