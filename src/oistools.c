#include "oistools.h"

double multiply_and_sum(size_t nsize, double *C1, double *C2);
double multiply_and_sum_mask(size_t nsize, double *C1, double *C2, char *mask);
void fill_c_matrices_for_kernel(int k_height, int k_width, int deg, int n,
                                int m, double *refimage, double *Conv);
void fill_c_matrices_for_background(int n, int m, int bkg_deg,
                                    double *Conv_bkg);

lin_system build_matrix_system(int n, int m, double *image, double *refimage,
                               int kernel_height, int kernel_width,
                               int kernel_polydeg, int bkg_deg, char *mask) {
  int kernel_size = kernel_height * kernel_width;
  int img_size = n * m;
  int kpdeg = kernel_polydeg;
  int poly_degree = (kpdeg + 1) * (kpdeg + 2) / 2;
  int bkg_dof;

  bkg_dof = (bkg_deg + 1) * (bkg_deg + 2) / 2;

  size_t conv_size = ((size_t)img_size) * (kernel_size * poly_degree + bkg_dof);
  double *Conv = calloc(conv_size, sizeof(*Conv)); // TODO err on bad calloc

  fill_c_matrices_for_kernel(kernel_height, kernel_width, kernel_polydeg, n, m,
                             refimage, Conv);
  double *Conv_bkg;
  if (bkg_deg != -1) {
    Conv_bkg = Conv + img_size * kernel_size * poly_degree;
    fill_c_matrices_for_background(n, m, bkg_deg, Conv_bkg);
  }

  // Create matrices M and vector b
  int total_dof = kernel_size * poly_degree + bkg_dof;
  size_t M_size = ((size_t)total_dof) * total_dof * sizeof(double);
  size_t b_size = ((size_t)total_dof) * sizeof(double);
  double *M = malloc(M_size);
  double *b = malloc(b_size);
  if (mask != NULL) {
    for (size_t i = 0; i < total_dof; i++) {
      double *C1 = Conv + i * img_size;
      for (size_t j = i; j < total_dof; j++) {
        double *C2 = Conv + j * img_size;
        M[i * total_dof + j] = multiply_and_sum_mask(img_size, C1, C2, mask);
        M[j * total_dof + i] = M[i * total_dof + j];
      }
      b[i] = multiply_and_sum_mask(img_size, image, C1, mask);
    }
  } else {
    for (size_t i = 0; i < total_dof; i++) {
      double *C1 = Conv + i * img_size;
      for (size_t j = i; j < total_dof; j++) {
        double *C2 = Conv + j * img_size;
        M[i * total_dof + j] = multiply_and_sum(img_size, C1, C2);
        M[j * total_dof + i] = M[i * total_dof + j];
      }
      b[i] = multiply_and_sum(img_size, image, C1);
    }
  }

  free(Conv);
  lin_system the_system = {total_dof, M, b};

  return the_system;
}

void convolve2d_adaptive(int n, int m, double *image, int kernel_height,
                         int kernel_width, int kernel_polydeg, double *kernel,
                         double *Conv) {
  // int k_side = kernel_height;
  int k_poly_dof = (kernel_polydeg + 1) * (kernel_polydeg + 2) / 2;

  for (long conv_row = 0; conv_row < n; ++conv_row) {
    for (long conv_col = 0; conv_col < m; ++conv_col) {
      int conv_index = conv_row * m + conv_col;

      for (int p = 0; p < kernel_height; p++) {
        for (int q = 0; q < kernel_width; q++) {
          long img_row =
              conv_row - (p - kernel_height / 2); // khs is kernel half side
          long img_col = conv_col - (q - kernel_width / 2);
          size_t img_index = img_row * m + img_col;

          // do only if img_index is in bounds of image
          if (img_row >= 0 && img_col >= 0 && img_row < n && img_col < m) {

            // reconstruct the (p, q) pixel of kernel
            double k_pixel = 0.0;
            // advance k_coeffs pointer to the p, q part
            double *k_coeffs_pq = kernel + (p * kernel_width + q) * k_poly_dof;
            size_t exp_index = 0;
            for (int exp_x = 0; exp_x <= kernel_polydeg; exp_x++) {
              for (int exp_y = 0; exp_y <= kernel_polydeg - exp_x; exp_y++) {
                k_pixel += k_coeffs_pq[exp_index] * pow(conv_row, exp_y) *
                           pow(conv_col, exp_x);
                exp_index++;
              }
            }

            Conv[conv_index] += image[img_index] * k_pixel;
          }
        }
      }

    } // conv_col
  }   // conv_row
}

double multiply_and_sum(size_t nsize, double *C1, double *C2) {
  double result = 0.0;
  for (size_t i = 0; i < nsize; i++) {
    result += C1[i] * C2[i];
  }
  return result;
}

double multiply_and_sum_mask(size_t nsize, double *C1, double *C2, char *mask) {
  double result = 0.0;
  for (size_t i = 0; i < nsize; i++) {
    if (mask[i] == 0)
      result += C1[i] * C2[i];
  }
  return result;
}

void fill_c_matrices_for_kernel(int k_height, int k_width, int deg, int n,
                                int m, double *refimage, double *Conv) {

  size_t img_size = n * m;
  int poly_degree = (deg + 1) * (deg + 2) / 2;

  for (size_t p = 0; p < k_height; p++) {
    for (size_t q = 0; q < k_width; q++) {
      double *Conv_pq = Conv + (p * k_width + q) * poly_degree * img_size;

      size_t exp_index = 0;
      for (int exp_x = 0; exp_x <= deg; exp_x++) {
        for (int exp_y = 0; exp_y <= deg - exp_x; exp_y++) {
          double *Conv_pqkl = Conv_pq + exp_index * img_size;

          for (long conv_row = 0; conv_row < n; ++conv_row) {
            for (long conv_col = 0; conv_col < m; ++conv_col) {
              size_t conv_index = conv_row * m + conv_col;
              long img_row =
                  conv_row - (p - k_height / 2); // khs is kernel half side
              long img_col = conv_col - (q - k_width / 2);
              size_t img_index = img_row * m + img_col;
              double x_pow = pow(conv_col, exp_x);
              double y_pow = pow(conv_row, exp_y);
              // make sure img_index is in bounds of refimage
              if (img_row >= 0 && img_col >= 0 && img_row < n && img_col < m) {
                Conv_pqkl[conv_index] = refimage[img_index] * x_pow * y_pow;
              }
            } // conv_col
          }   // conv_row

          exp_index++;
        } // exp_y
      }   // exp_x

    } // q
  }   // p

  return;
}

void fill_c_matrices_for_background(int n, int m, int bkg_deg,
                                    double *Conv_bkg) {

  int exp_index = 0;
  for (size_t exp_x = 0; exp_x <= bkg_deg; exp_x++) {
    for (size_t exp_y = 0; exp_y <= bkg_deg - exp_x; exp_y++) {

      double *Conv_xy = Conv_bkg + exp_index * n * m;

      for (long conv_row = 0; conv_row < n; ++conv_row) {
        for (long conv_col = 0; conv_col < m; ++conv_col) {
          size_t conv_index = conv_row * m + conv_col;
          double x_pow = pow(conv_col, exp_x);
          double y_pow = pow(conv_row, exp_y);

          Conv_xy[conv_index] = x_pow * y_pow;
        } // conv_col
      }   // conv_row

      exp_index++;
    } // exp_y
  }   // exp_x

  return;
}
