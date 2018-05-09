#include <stdio.h>
#include "test_ois_tools.h"

int main(int argc, const char * argv[]) {
    
    simple_convolve2d_adaptive_run();
    printf("Finished\n");
    
    return EXIT_SUCCESS;
}
