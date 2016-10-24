#include <Python.h>
//#include <numpy/npy_common.h>
//#include <numpy/ndarrayobject.h>
#include <numpy/arrayobject.h>

double multiply_and_sum(int nsize, double* C1, double* C2);
double multiply_and_sum_mask(int nsize, double* C1, double* C2, char* mask);
void fill_c_matrices_for_kernel(int k_side, int deg, int n, int m, double* refimage, double* Conv);
void fill_c_matrices_for_background(int n, int m, int bkg_deg, double* Conv_bkg);


static PyObject *
varconv_gen_matrix_system(PyObject *self, PyObject *args)
{
    PyArrayObject *np_image, *np_refimage, *np_mask;
    int k_side;
    int deg; // The degree of the varying polynomial for the kernel
    int bkg_deg; // The degree of the varying polynomial for the background
    unsigned char hasmask;

    if (!PyArg_ParseTuple(args, "O!O!bOiii", &PyArray_Type, &np_image,
            &PyArray_Type, &np_refimage, &hasmask, &np_mask,
            &k_side, &deg, &bkg_deg)) {
        return NULL;
    }
    if (NULL == np_image) return NULL;
    if (NULL == np_refimage) return NULL;
    if (NULL == np_mask) return NULL;

    int n = np_image->dimensions[0];
    int m = np_image->dimensions[1];

    double* image = (double*)np_image->data;
    double* refimage = (double*)np_refimage->data;
    char* mask;
    if (hasmask == 1) {
        mask = (char*)np_mask->data;
    } else {
        mask = NULL;
    }
    
    int kernel_size = k_side * k_side;
    int img_size = n * m;
    int poly_degree = (deg + 1) * (deg + 2) / 2;
    int bkg_dof;
    if (bkg_deg == -1) {
        bkg_dof = 0;
    } else {
        bkg_dof = (bkg_deg + 1) * (bkg_deg + 2) / 2;
    }
    double* Conv = calloc(img_size * (kernel_size * poly_degree + bkg_dof), sizeof(*Conv));
    fill_c_matrices_for_kernel(k_side, deg, n, m, refimage, Conv);
    double* Conv_bkg;
    if (bkg_deg != -1) {
        Conv_bkg = Conv + img_size * kernel_size * poly_degree;
        fill_c_matrices_for_background(n, m, bkg_deg, Conv_bkg);
    }

    //Create matrices M and vector b
    int total_dof = kernel_size * poly_degree + bkg_dof;
    double* M = malloc(total_dof * total_dof * sizeof(*M));
    double* b = malloc(total_dof * sizeof(*b));
    if (hasmask == 1) {
        for (int i = 0; i < total_dof; i++) {
            double* C1 = Conv + i * img_size;
            for (int j = i; j < total_dof; j++) {
                double* C2 = Conv + j * img_size;
                M[i * total_dof + j] = multiply_and_sum_mask(img_size, C1, C2, mask);
                M[j * total_dof + i] = M[i * total_dof + j];
            }
            b[i] = multiply_and_sum_mask(img_size, image, C1, mask);
        }
    } else {
        for (int i = 0; i < total_dof; i++) {
            double* C1 = Conv + i * img_size;
            for (int j = i; j < total_dof; j++) {
                double* C2 = Conv + j * img_size;
                M[i * total_dof + j] = multiply_and_sum(img_size, C1, C2);
                M[j * total_dof + i] = M[i * total_dof + j];
            }
            b[i] = multiply_and_sum(img_size, image, C1);
        }
    }

    //free(Conv);

    npy_intp Mdims[2] = {total_dof, total_dof};
    npy_intp bdims = total_dof;
    npy_intp convdims[4] = {kernel_size, poly_degree, bkg_dof, img_size};
    PyObject* pyM = PyArray_SimpleNewFromData(2, Mdims, NPY_DOUBLE, M);
    PyObject* pyb = PyArray_SimpleNewFromData(1, &bdims, NPY_DOUBLE, b);
    PyObject* pyConv = PyArray_SimpleNewFromData(4, convdims, NPY_DOUBLE, Conv);

    return Py_BuildValue("OOO", pyM, pyb, pyConv);
}


static PyObject *
varconv_convolve2d_adaptive(PyObject *self, PyObject *args) {
    PyArrayObject *np_image, *np_kernelcoeffs;
    int deg; // The degree of the varying polynomial

    if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &np_image,
            &PyArray_Type, &np_kernelcoeffs, &deg))  return NULL;
    if (NULL == np_image) return NULL;
    if (NULL == np_kernelcoeffs) return NULL;

    int n = np_image->dimensions[0];
    int m = np_image->dimensions[1];
    int k_side = np_kernelcoeffs->dimensions[0];
    int k_poly_dof = np_kernelcoeffs->dimensions[2];
    int khs = k_side / 2;

    double* image = (double*)np_image->data;
    double* k_coeffs = (double*)np_kernelcoeffs->data;
    double k_pixel;

    double* Conv = (double*)calloc(n * m, sizeof(*Conv));

    for (int conv_row = 0; conv_row < n; ++conv_row) {
        for (int conv_col = 0; conv_col < m; ++conv_col) {
            int conv_index = conv_row * m + conv_col;

            for (int p = 0; p < k_side; p++) {
                for (int q = 0; q < k_side; q++) {
                    int img_row = conv_row - (p - khs); // khs is kernel half side
                    int img_col = conv_col - (q - khs);
                    int img_index = img_row * m + img_col;

                    // do only if img_index is in bounds of image
                    if (img_row >= 0 && img_col >=0 && img_row < n && img_col < m) {

                        // reconstruct the (p, q) pixel of kernel
                        k_pixel = 0.0;
                        // advance k_coeffs pointer to the p, q part
                        double* k_coeffs_pq = k_coeffs + (p * k_side + q) * k_poly_dof;
                        int exp_index = 0;
                        for (int exp_x = 0; exp_x <= deg; exp_x++) {
                            for (int exp_y = 0; exp_y <= deg - exp_x; exp_y++) {
                                k_pixel += k_coeffs_pq[exp_index] * pow(conv_row, exp_y) * pow(conv_col, exp_x);
                                exp_index++;
                            }
                        }

                        Conv[conv_index] += image[img_index] * k_pixel;
                    }
                }
            }

        } // conv_col
    } // conv_row

    npy_intp Convdims[2] = {n, m};
    PyObject* pyConv = PyArray_SimpleNewFromData(2, Convdims, NPY_DOUBLE, Conv);

    return Py_BuildValue("O", pyConv);
    //return pyConv;

}

static PyMethodDef VarConvMethods[] = {
    {"gen_matrix_system", varconv_gen_matrix_system, METH_VARARGS,
     "Generate the matrix system to find best convolution parameters."},
    {"convolve2d_adaptive", varconv_convolve2d_adaptive, METH_VARARGS,
     "Convolves image with an adaptive kernel."},
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


double multiply_and_sum_mask(int nsize, double* C1, double* C2, char* mask) {
    double result = 0.0;
    for (int i = 0; i < nsize; i++) {
        if (mask[i] == 0) result += C1[i] * C2[i];
    }
    return result;
}

void fill_c_matrices_for_kernel(int k_side, int deg, int n, int m, double* refimage, double* Conv) {

    int img_size = n * m;
    int khs = k_side / 2; // kernel half side
    int poly_degree = (deg + 1) * (deg + 2) / 2;

    for (int p = 0; p < k_side; p++) {
        for (int q = 0; q < k_side; q++) {
            double* Conv_pq = Conv + (p * k_side + q) * poly_degree * img_size;

            int exp_index = 0;
            for (int exp_x = 0; exp_x <= deg; exp_x++) {
                for (int exp_y = 0; exp_y <= deg - exp_x; exp_y++) {
                    double* Conv_pqkl = Conv_pq + exp_index * img_size;
                    
                    for (int conv_row = 0; conv_row < n; ++conv_row) {
                        for (int conv_col = 0; conv_col < m; ++conv_col) {
                            int conv_index = conv_row * m + conv_col;
                            int img_row = conv_row - (p - khs); // khs is kernel half side
                            int img_col = conv_col - (q - khs);
                            int img_index = img_row * m + img_col;
                            double x_pow = pow(conv_col, exp_x);
                            double y_pow = pow(conv_row, exp_y);
                            // make sure img_index is in bounds of refimage
                            if (img_row >= 0 && img_col >=0 && img_row < n && img_col < m) {
                                Conv_pqkl[conv_index] = refimage[img_index] * x_pow * y_pow;
                            }
                        } // conv_col
                    } // conv_row
                    
                    exp_index++;
                } // exp_y
            } // exp_x
            
        } //q
    } // p

    return;
}

void fill_c_matrices_for_background(int n, int m, int bkg_deg, double* Conv_bkg) {

    int exp_index = 0;
    for (int exp_x = 0; exp_x <= bkg_deg; exp_x++) {
        for (int exp_y = 0; exp_y <= bkg_deg - exp_x; exp_y++) {

            double* Conv_xy = Conv_bkg + exp_index * n * m;
            
            for (int conv_row = 0; conv_row < n; ++conv_row) {
                for (int conv_col = 0; conv_col < m; ++conv_col) {
                    int conv_index = conv_row * m + conv_col;
                    double x_pow = pow(conv_col, exp_x);
                    double y_pow = pow(conv_row, exp_y);

                    Conv_xy[conv_index] = x_pow * y_pow;
                } // conv_col
            } // conv_row
            
            exp_index++;
        } // exp_y
    } // exp_x

    return;   
}
