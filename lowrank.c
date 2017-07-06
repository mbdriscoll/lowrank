#include <stdlib.h>
#include <string.h>

#include <omp.h>

#include "lowrank.h"

#define MKL_Complex8  complex float
#define MKL_Complex16 complex double

#include <mkl.h>
#include <mkl_types.h>

#define min(x,y) (((x)<(y))?(x):(y))
#define max(x,y) (((x)>(y))?(x):(y))

// macro for indexing into a block
#define IDX_BLK(t0,t1,bx,by,bz) \
   ((t0) * (T1*b*b*b*1) + \
    (t1) * (   b*b*b*1) + \
    (bx) * (     b*b*1) + \
    (by) * (       b*1) + \
    (bz) * (         1))

// macros for indexing into an img
#define IDX_IMG(t0,w,t1,x,y,z) \
  ((t0) * ( W*T1*X*Y*Z*1) + \
   ( w) * (   T1*X*Y*Z*1) + \
   (t1) * (      X*Y*Z*1) + \
   ( x) * (        Y*Z*1) + \
   ( y) * (          Z*1) + \
   ( z) * (            1))

void svthresh(
    float thresh, complex float *imgs,
    int T0, // outer time dimension
    int W,  // image sets
    int T1, // inner time dimension
    int X, int Y, int Z, // image dimensions
    int b, int sx, int sy, int sz) // block size, shifts
{
    int M = b*b*b,
        N = T0*T1;
    int K = min(M,N);
    int ldc  = M,
        ldu  = M,
        ldvh = K;

    #pragma omp parallel
    {
        const complex float alpha = 1.0, beta = 0.0;
        complex float *U     = malloc( M * K * sizeof(complex float) );
        complex float *VH    = malloc( N * K * sizeof(complex float) );
        float         *s     = malloc(     K * sizeof(        float) );
        float    *superb     = malloc(     K * sizeof(        float) );
        complex float *block = malloc( M * N * sizeof(complex float) );

        // for every block...
        #pragma omp for collapse(4)
        for (int w =   0; w < W; w++) {
        for (int x = -sx; x < X; x += b) {
        for (int y = -sy; y < Y; y += b) {
        for (int z = -sz; z < Z; z += b) {

            // block[:] = 0
            memset(block, 0, M * N * sizeof(complex float));

            // load block
            for (int t0 = 0; t0 < T0; t0++) {
            for (int t1 = 0; t1 < T1; t1++) {
            for (int bx = 0; bx <  b; bx++) { if (0 <= x+bx && x+bx < X) {
            for (int by = 0; by <  b; by++) { if (0 <= y+by && y+by < Y) {
            for (int bz = 0; bz <  b; bz++) { if (0 <= z+bz && z+bz < Z) {
                  block[ IDX_BLK(t0,t1,bx,by,bz) ] = imgs[ IDX_IMG(t0,w,t1,x+bx,y+by,z+bz) ];
            }}}}}}}}

            // U, s, VH = svd(block)
            LAPACKE_cgesvd( LAPACK_COL_MAJOR, 'S', 'S',
                M, N, block, ldc, s, U, ldu, VH, ldvh, superb );

            // sV = thresh(s) * VH
            for (int n = 0; n < N; n++)
                for (int k = 0; k < K; k++)
                    VH[n*K+k] *= max(s[k]-thresh, 0);

            // block = U * sV
            cblas_cgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                 M, N, K, &alpha, U, ldu, VH, ldvh, &beta, block, ldc );

            // restore block
            for (int t0 = 0; t0 < T0; t0++) {
            for (int t1 = 0; t1 < T1; t1++) {
            for (int bx = 0; bx <  b; bx++) { if (0 <= x+bx && x+bx < X) {
            for (int by = 0; by <  b; by++) { if (0 <= y+by && y+by < Y) {
            for (int bz = 0; bz <  b; bz++) { if (0 <= z+bz && z+bz < Z) {
                  imgs[ IDX_IMG(t0,w,t1,x+bx,y+by,z+bz) ] = block[ IDX_BLK(t0,t1,bx,by,bz) ];
            }}}}}}}}

        }}}}

        free(block); free(U); free(VH); free(s); free(superb);
    }
}
