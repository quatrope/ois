#include <stdio.h>
#include "test_ois_tools.h"

int main(int argc, const char * argv[]) {
    printf("\nStarting Tests:\n");
    printf("Running simple_convolve2d_adaptive_run test...");
    simple_convolve2d_adaptive_run();
    printf("ok\n");
    printf("Running simple_build_matrix_system_run test...");
    simple_build_matrix_system_run();
    printf("ok\n");
    printf("Finished\n");
    
    return EXIT_SUCCESS;
}
