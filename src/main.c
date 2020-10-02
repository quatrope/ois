/*****************************************************************************
 * Driver main file for ois as a stand-alone C progrmam.                     *
 *                                                                           *
 * (c) Martin Beroiz                                                         *
 * martinberoiz@gmail.com                                                    *
 *****************************************************************************/

#include "fitshelper.h"
#include "fitsio.h"
#include "oistools.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void perform_subtraction(image sciimg, image refimg, int kernel_height,
                         int kernel_width, int kernel_polydeg, double *kernel,
                         double *diff_data);
void solve_linear_system(int n, double *C, double *D, double *xcs);

char version_number[] = "1.0";

void usage(char *exec_name);
void version(char *exec_name);

int main(int argc, char *argv[]) {
  // Start the clock
  clock_t begin = clock();
  char *exec_name = argv[0];

  // Parse command line arguments:
  // Default arguments
  int kside = -1; // The half-width of the kernel side
  int kdeg =
      -1; // Degree of the interpolating polynomial for the variable kernel
  char *reffile = NULL;
  char *scifile = NULL;
  char refstarsfile_default[] = "./refstars.txt";
  char *refstarsfile = refstarsfile_default;
  char outputfile_default[] = "diff_img.fits";
  char *outputfile = outputfile_default;
  if (argc < 2) {
    usage(exec_name);
    return EXIT_SUCCESS;
  } else {
    ++argv; // Skip the invocation program name
    --argc;
    while (argc > 0) {
      if (!strcmp(*argv, "-ks") || !strcmp(*argv, "--kernel-side")) {
        ++argv; // Consume one word
        --argc; // Decrease word counter by 1
        kside = atoi(*argv);
      } else if (!strcmp(*argv, "-ref")) {
        ++argv; // Consume one word
        --argc; // Decrease word counter by 1
        reffile = *argv;
      } else if (!strcmp(*argv, "-sci")) {
        ++argv; // Consume one word
        --argc; // Decrease word counter by 1
        scifile = *argv;
      } else if (!strcmp(*argv, "-kd") || !strcmp(*argv, "--kernel-poly-deg")) {
        ++argv; // Consume one word
        --argc; // Decrease word counter by 1
        kdeg = atoi(*argv);
      } else if (!strcmp(*argv, "-o")) {
        ++argv; // Consume one word
        --argc; // Decrease word counter by 1
        outputfile = *argv;
      } else if (!strcmp(*argv, "-h") || !strcmp(*argv, "--help")) {
        usage(exec_name);
        return EXIT_SUCCESS;
      } else if (!strcmp(*argv, "--version")) {
        version(exec_name);
        return EXIT_SUCCESS;
      } else {
        printf("Unexpected Argument: %s\n", *argv);
        usage(exec_name);
        return EXIT_FAILURE;
      }
      ++argv;
      --argc;
    }
  }
  // Check here which variables were not set
  if (kdeg == -1) {
    printf("Undefined value for -kd (kernel polynomial deg). Exiting.\n");
    usage(exec_name);
    return EXIT_FAILURE;
  }
  if (kside == -1) {
    printf("Undefined value for -ks (kernel side). Exiting.\n");
    usage(exec_name);
    return EXIT_FAILURE;
  }
  if (kside % 2 == 0) {
    printf("Kernel side must be an odd number. Exiting.\n");
    return EXIT_FAILURE;
  }

  // Open and read fits files
  image refimg = fits_get_data(reffile);
  double *Ref = refimg.data;
  image sciimg = fits_get_data(scifile);
  double *Sci = sciimg.data;

  // Put a guard in case images are of different shape
  if (refimg.n != sciimg.n || refimg.m != sciimg.m) {
    printf("ERROR: Reference and Science images have different dimensions.\n");
    return EXIT_FAILURE;
  }

  // Here do subtraction
  double *diff_data = (double *)malloc(sizeof(double) * sciimg.n * sciimg.m);
  int kpoly_dof = (kdeg + 1) * (kdeg + 2) / 2;
  // The array total length for the kernel
  int klen = kside * kside * kpoly_dof;
  double *kernel = (double *)malloc(sizeof(double) * klen);
  perform_subtraction(sciimg, refimg, kside, kside, kdeg, kernel, diff_data);
  free(Ref);
  free(Sci);

  image diffimg = {diff_data, sciimg.n, sciimg.m};
  int success = fits_write_to(outputfile, diffimg, scifile);
  free(diff_data);
  if (success == EXIT_FAILURE) {
    printf("Problem writing diff FITS file.\n");
    return EXIT_FAILURE;
  }
  // Do something with kernel?
  free(kernel);

  clock_t end = clock();
  printf("The difference took %f seconds\n",
         (double)(end - begin) / CLOCKS_PER_SEC);
}

void perform_subtraction(image sciimg, image refimg, int kernel_height,
                         int kernel_width, int kernel_polydeg, double *kernel,
                         double *diff_data) {

  int bkg_deg = -1;  // Don't fit background
  char *mask = NULL; // No masked pixels
  int n = sciimg.n;
  int m = sciimg.m;
  // Create the linar matrix system to solve for kernel
  lin_system result_sys =
      build_matrix_system(n, m, sciimg.data, refimg.data, kernel_height,
                          kernel_width, kernel_polydeg, bkg_deg, mask);

  // Get kernel
  // self.coeffs = np.linalg.solve(m, b)
  solve_linear_system(result_sys.b_dim, result_sys.M, result_sys.b, kernel);

  // Get opt_img by convolving ref_img with kernel
  // opt_image = signal.convolve2d(self.refimage, self.get_kernel(),
  // mode="same")
  double *opt_data = (double *)malloc(n * m * sizeof(*opt_data));
  convolve2d_adaptive(n, m, refimg.data, kernel_height, kernel_width,
                      kernel_polydeg, kernel, opt_data);

  // subtract image - opt_image
  for (int i = 0; i < n * m; ++i) {
    diff_data[i] = sciimg.data[i] - opt_data[i];
  }
}

void solve_linear_system(int n, double *C, double *D, double *xcs) {
  /** Solves the linear algebraic matrix system  */
  int nsq = n * n;
  double *Low = (double *)calloc(sizeof(double), nsq);
  double *U = (double *)calloc(sizeof(double), nsq);

  // Now we need to do the LU decomposition
  for (int k = 0; k < n; k++) {
    Low[k + k * n] = 1.0;
    for (int i = k + 1; i < n; i++) {
      Low[k + i * n] = C[k + i * n] / C[k + k * n];
      for (int j = k + 1; j < n; j++) {
        C[j + i * n] = C[j + i * n] - Low[k + i * n] * C[j + k * n];
      }
    }
    for (int j = k; j < n; j++) {
      U[j + k * n] = C[k + j * n];
    }
  }

  // Now we will do Gaussian elimination
  // Solve for xc
  double *ycs = (double *)calloc(sizeof(double), n);
  for (int i = 0; i < (n - 1); i++) {
    for (int j = (i + 1); j < n; j++) {
      double ratio = Low[j + i * n] / Low[i + i * n];
      for (int count = i; count < n; count++) {
        Low[count + j * n] -= ratio * Low[count + i * count];
      }
      D[j] -= (ratio * D[i]);
    }
  }
  ycs[n - 1] = D[n - 1] / Low[(n - 1) + n * (n - 1)];
  for (int i = (n - 2); i >= 0; i--) {
    double temp = D[i];
    for (int j = (i + 1); j < n; j++) {
      temp -= (Low[j + i * n] * ycs[j]);
    }
    ycs[i] = temp / Low[i + i * n];
  }

  // Solve for xc
  for (int i = 0; i < (n - 1); i++) {
    for (int j = (i + 1); j < n; j++) {
      double ratio = U[j + i * n] / U[i + i * n];
      for (int count = i; count < n; count++) {
        U[count + j * n] -= ratio * Low[count + i * count];
      }
      ycs[j] -= (ratio * ycs[i]);
    }
  }
  xcs[n - 1] = ycs[n - 1] / U[(n - 1) + n * (n - 1)];
  for (int i = (n - 2); i >= 0; i--) {
    double temp = ycs[i];
    for (int j = (i + 1); j < n; j++) {
      temp -= U[j + i * n] * xcs[j];
    }
    xcs[i] = temp / U[i + i * n];
  }

  // free everything//
  free(ycs);
  free(U);
  free(Low);
}

void usage(char *exec_name) {
  char *exec_basename = strrchr(exec_name, '/') + 1;
  if (exec_basename == NULL)
    exec_basename = exec_name;
  printf("%s\nAuthor: Martin Beroiz (c)\n", exec_basename);
  printf("------------------------\n\n");
  printf(
      "usage: %s -ks, --kernel-side <int> -kd, --kernel-poly-deg <int> -ref "
      "<filename> -sci <filename> [-o <filename>] [-h, --help] [--version]\n\n",
      exec_basename);
  printf("Arguments:\n");
  printf("\t-ks, --kernel-side: the side in pixels of the kernel to calculate "
         "the optimal difference. Must be an odd number.\n");
  printf("\t-kd, --kernel-poly-deg: Degree of the interpolating polynomial for "
         "the variable kernel.\n");
  printf("\t-ref: The reference image path.\n");
  printf("\t-sci: The science image path.\n");
  printf("\t-o [optional]: The path to the subtraction fits file.\n");
  printf("\t\tDefault value is \"diff_img.fits\".\n");
  printf("\t-h, --help: Print this help and exit.\n");
  printf("\t--version: Print version information and exit.\n");
  printf("\n");
}

void version(char *exec_name) {
  char *exec_basename = strrchr(exec_name, '/') + 1;
  if (exec_basename == NULL)
    exec_basename = exec_name;
  printf("%s %s\n", exec_basename, version_number);
}