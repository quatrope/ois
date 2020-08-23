#include "fitshelper.h"

image fits_get_data(char *filename) {
  fitsfile *fp;
  int status = 0;
  int bitpix, naxis;
  long naxes[2];
  fits_open_file(&fp, filename, READONLY, &status);
  fits_get_img_param(fp, 2, &bitpix, &naxis, naxes, &status);
  int N = (int)(naxes[0] * naxes[1]); // size of the image //
  long initial_pixel[] = {1, 1};
  double *data = (double *)calloc(N, sizeof(double));
  fits_read_pix(fp, TDOUBLE, initial_pixel, N, 0, data, 0, &status);
  fits_close_file(fp, &status);
  image img = {data, (int)naxes[0], (int)naxes[1]};
  return img;
}

int fits_write_to(char *filename, image img, char *file_with_header) {
  long nax[] = {img.n, img.m};
  long initial_pixel[] = {1, 1};
  int status = 0;
  fitsfile *fp;

  fits_create_file(&fp, filename, &status);
  fits_create_img(fp, DOUBLE_IMG, 2, nax, &status);
  double *transpose = (double *)malloc(img.n * img.m * sizeof(double));
  for (int i = 0; i < img.m; i++) {
    for (int j = 0; j < img.n; j++) {
      transpose[j + i * img.n] = img.data[i + j * img.n];
    }
  }
  fits_write_pix(fp, TDOUBLE, initial_pixel, img.n * img.m, transpose, &status);
  free(transpose);

  // Set the Header on the new image
  fitsfile *headfp;
  fits_open_file(&headfp, file_with_header, READWRITE, &status);
  int nkeys;
  fits_get_hdrspace(headfp, &nkeys, NULL, &status);
  char card[FLEN_CARD];
  for (int i = 11; i < nkeys; i++) {
    fits_read_record(headfp, i, card, &status);
    fits_write_record(fp, card, &status);
  }

  fits_close_file(headfp, &status);
  fits_close_file(fp, &status);

  return EXIT_SUCCESS;
}
