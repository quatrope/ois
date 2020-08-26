#include "fitshelper.h"
#include <string.h>

image fits_get_data(char *filename) {
  fitsfile *fp;
  int status = 0;
  char err_msg[32];
  image badimage = {NULL, 0, 0};
  int bitpix, naxis;
  long naxes[2];
  fits_open_file(&fp, filename, READONLY, &status);
  if (status) {
    printf("Error opening file %s\n", filename);
    fits_get_errstatus(status, err_msg);
    printf("ERROR: %s\n", err_msg);
    return badimage;
  }
  fits_get_img_param(fp, 2, &bitpix, &naxis, naxes, &status);
  if (status) {
    printf("Error getting parameters for file %s\n", filename);
    fits_get_errstatus(status, err_msg);
    printf("ERROR: %s\n", err_msg);
    return badimage;
  }
  int N = (int)(naxes[0] * naxes[1]); // size of the image //
  long initial_pixel[] = {1, 1};
  double *data = (double *)calloc(N, sizeof(double));
  fits_read_pix(fp, TDOUBLE, initial_pixel, N, 0, data, 0, &status);
  if (status) {
    printf("Error getting pixels from file %s\n", filename);
    fits_get_errstatus(status, err_msg);
    printf("ERROR: %s\n", err_msg);
    return badimage;
  }
  fits_close_file(fp, &status);
  image img = {data, (int)naxes[0], (int)naxes[1]};
  return img;
}

int fits_write_to(char *filename, image img, char *file_with_header) {
  long nax[] = {img.n, img.m};
  long initial_pixel[] = {1, 1};
  int status = 0;
  char err_msg[32];
  fitsfile *fp;
  char *filename_overwrite =
      (char *)malloc((strlen(filename) + 2) * sizeof(char));
  sprintf(filename_overwrite, "%s", filename);
  fits_create_file(&fp, filename_overwrite, &status);
  if (status) {
    printf("Error creating  file %s %s\n", filename_overwrite, filename);
    fits_get_errstatus(status, err_msg);
    printf("ERROR: %s\n", err_msg);
    return EXIT_FAILURE;
  }
  fits_create_img(fp, DOUBLE_IMG, 2, nax, &status);
  if (status) {
    printf("Error creating  image array for file %s\n", filename);
    fits_get_errstatus(status, err_msg);
    printf("ERROR: %s\n", err_msg);
    return EXIT_FAILURE;
  }
  double *transpose = (double *)malloc(img.n * img.m * sizeof(double));
  for (int i = 0; i < img.m; i++) {
    for (int j = 0; j < img.n; j++) {
      transpose[j + i * img.n] = img.data[i + j * img.n];
    }
  }
  fits_write_pix(fp, TDOUBLE, initial_pixel, img.n * img.m, transpose, &status);
  free(transpose);
  if (status) {
    printf("Error writing pixels for file %s\n", filename);
    fits_get_errstatus(status, err_msg);
    printf("ERROR: %s\n", err_msg);
    return EXIT_FAILURE;
  }

  // Set the Header on the new image
  fitsfile *headfp;
  fits_open_file(&headfp, file_with_header, READWRITE, &status);
  if (status) {
    printf("Error opening file %s\n", file_with_header);
    fits_get_errstatus(status, err_msg);
    printf("ERROR: %s\n", err_msg);
    return EXIT_FAILURE;
  }
  int nkeys;
  fits_get_hdrspace(headfp, &nkeys, NULL, &status);
  if (status) {
    printf("Error getting header for file %s\n", file_with_header);
    fits_get_errstatus(status, err_msg);
    printf("ERROR: %s\n", err_msg);
    return EXIT_FAILURE;
  }
  char card[FLEN_CARD];
  for (int i = 11; i < nkeys; i++) {
    fits_read_record(headfp, i, card, &status);
    fits_write_record(fp, card, &status);
  }

  fits_close_file(headfp, &status);
  fits_close_file(fp, &status);
  return EXIT_SUCCESS;
}
