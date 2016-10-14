#include <Python.h>
//#include <numpy/npy_common.h>
#include <numpy/ndarrayobject.h>
#include <numpy/arrayobject.h>

double multiply_and_sum(int nsize, double* C1, double* C2);

static PyObject *
varconv_cconvolve_var(PyObject *self, PyObject *args)
{
    PyArrayObject *np_image, *np_refimage;
    int kernel_width, kernel_height;
    int deg; // The degree of the varying polynomial

    if (!PyArg_ParseTuple(args, "O!O!iii", &PyArray_Type, &np_image,
            &PyArray_Type, &np_refimage, &kernel_width, &kernel_height, &deg))  return NULL;
    if (NULL == np_image)  return NULL;
    if (NULL == np_refimage)  return NULL;

    int n = np_image->dimensions[0];
    int m = np_image->dimensions[1];

    double* image = (double*)np_image->data;
    double* refimage = (double*)np_refimage->data;

    double* Conv = (double*)malloc(n * m * sizeof(*Conv));

    int kernel_size = kernel_height * kernel_width;
    int img_size = n * m;
    int poly_degree = (deg + 1) * (deg + 2) / 2;
    Conv = calloc(img_size * kernel_size * poly_degree, sizeof(*Conv));

    for (int p = 0; p < kernel_height; ++p) {
        for (int q = 0; q < kernel_width; ++q) {
            double* Conv_pq = Conv + p * kernel_width + q;

            int exp_index = 0;
            for (int exp_x = 0; exp_x <= deg; exp_x++) {
                double p_pow = pow(p, exp_x);
                for (int exp_y = 0; exp_y <= deg - exp_x; exp_y++) {
                    double q_pow = pow(q, exp_y);
                    double* Conv_pqkl = Conv_pq + exp_index;

                    for (int conv_row = 0; conv_row < n; ++conv_row) {
                        for (int conv_col = 0; conv_col < m; ++conv_col) {
                            int conv_index = conv_row * m + conv_col;
                            int img_index = conv_index - p * m - q;
                            // make sure img_index is in bounds of refimage
                            if (img_index >= 0 && img_index < img_size) {
                                Conv_pqkl[conv_index] = refimage[img_index] * p_pow * q_pow;
                            }
                        } // conv_col
                    } // conv_row

                    exp_index++;                    
                } // exp_y
            } // exp_x

        } //q
    } // p

    //Create matrices M and vector b
    int total_dof = kernel_size * poly_degree;
    double* Mb = malloc(total_dof * (total_dof + 1) * sizeof(*Mb));
    //double* b = malloc(total_dof * sizeof(*b));
    for (int i = 0; i < total_dof; i++) {
        double* C1 = Conv + i * img_size;
        for (int j = i; j < total_dof; j++) {
            double* C2 = Conv + j * img_size;
            Mb[i * total_dof + j] = multiply_and_sum(img_size, C1, C2);
            Mb[j * total_dof + i] = Mb[i*total_dof + j];
        }
        Mb[total_dof * total_dof + i] = multiply_and_sum(img_size, image, C1);
    }

    free(Conv);

    npy_intp dims[2] = {total_dof + 1, total_dof};
    PyObject* pyM = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, Mb);

    return pyM;
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

double multiply_and_sum(int nsize, double* C1, double* C2) {
    double result = 0.0;
    for (int i = 0; i < nsize; i++) {
        result += C1[i] * C2[i];
    }
    return result;
}
