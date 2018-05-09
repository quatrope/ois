#include "test_ois_tools.h"

int simple_convolve2d_adaptive_run() {
    double* solution;
    int n = 10;
    int m = 20;
    double* image = (double *)malloc(n * m * sizeof(double));
    int kernel_height = 3;
    int kernel_width = 5;
    int kernel_polydeg = 1;
    int k_polydof = (kernel_polydeg + 1) * (kernel_polydeg + 2) / 2;
    double * conv = (double *)malloc(n * m * sizeof(double));
    double* kernel = (double*)malloc(kernel_height * kernel_width * k_polydof * sizeof(double));
    solution = convolve2d_adaptive(n, m, image, kernel_height, kernel_width, kernel_polydeg, kernel, conv);
    return EXIT_SUCCESS;
}

int simple_build_matrix_system_run() {

    int n = 10;
    int m = 20;
    double* image = (double *)malloc(n * m * sizeof(double));
    double* refimage = (double *)malloc(n * m * sizeof(double));
    int kernel_height = 3;
    int kernel_width = 5;
    int kernel_polydeg = 1;
    int bkg_deg = 2;
    
    build_matrix_system(n, m, image, refimage, kernel_height, kernel_width,
                                         kernel_polydeg, bkg_deg, NULL);
    return EXIT_SUCCESS;
}
