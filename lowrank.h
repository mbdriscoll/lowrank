#include <complex.h>

void svthresh(
    float thresh,                // threshold
    int block_size,              // block size
    int sx, int sy, int sz,      // shift values
    int T0, int P, int T1,       // time, space, time dims
    int X, int Y, int Z,         // img dims
    complex float *imgs          // data to threshold
);
