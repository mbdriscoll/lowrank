#include <complex.h>

void svthresh(
    float thresh, complex float *imgs,
    int T0, int W, int T1, int X, int Y, int Z, // image dimensions
    int b, int sx, int sy, int sz // block size, shifts
);
