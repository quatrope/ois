#ifndef fitshelper_h
#define fitshelper_h

#include <stdio.h>
#include "fitsio.h"

typedef struct {
    double* data;
    int n;
    int m;
} image;

image fits_get_data(char* filename);
int fits_write_to(char* filename, image img, char* header_file);

#endif
