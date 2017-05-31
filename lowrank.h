#include <complex.h>

void svthresh(
    float thresh,                // threshold
    int block_size,              // block size
    int sx, int sy, int sz,      // shift values
    int T, int X, int Y, int Z,  // dimensions of imgs
    complex float *imgs          // data to threshold
);
