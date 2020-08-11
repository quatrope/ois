#include <math.h>
#include <stdlib.h>

typedef struct {
  int b_dim;
  double *M;
  double *b;
} lin_system;

lin_system build_matrix_system(int n, int m, double *image, double *refimage,
                               int kernel_height, int kernel_width,
                               int kernel_polydeg, int bkg_deg, char *mask);

void convolve2d_adaptive(int n, int m, double *image, int kernel_height,
                         int kernel_width, int kernel_polydeg, double *kernel,
                         double *convolution);
