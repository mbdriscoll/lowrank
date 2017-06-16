#include <stdlib.h>
#include <string.h>

#include "lowrank.h"

#define MKL_Complex8  complex float
#define MKL_Complex16 complex double

#include <mkl.h>
#include <mkl_types.h>

#define min(x,y) (((x)<(y))?(x):(y))
#define max(x,y) (((x)>(y))?(x):(y))

// indexing macros
#define IDX_BLK(t0,t1,x,y,z) ((t0)*T1*b*b*b + (t1)*b*b*b + (x)*b*b + (y)*b + (z))
#define IDX_IMG(t0,p,t1,x,y,z) ((t0)*C*T1*b*b*b + (p)*T1*b*b*b + (t1)*b*b*b + (x)*b*b + (y)*b + (z))

void svthresh(float thresh, int b, // threshold and block size
    int sx, int sy, int sz,        // shift values
    int T0, int C, int T1,         // time/blockid/time dimensions
     int X, int Y, int Z,          // dimensions of imgs
    complex float *imgs)           // data to threshold
{
    int M = b*b*b,
        N = T0*T1;
    int K = min(M,N);
    int ldc = M,
        ldu = M,
        ldv = K;

    #pragma omp parallel
    {
        const complex float alpha = 1.0, beta = 0.0;
        complex float *U     = malloc( M * K * sizeof(complex float) );
        complex float *V     = malloc( K * N * sizeof(complex float) );
        float         *s     = malloc(     K * sizeof(        float) );
        float    *superb     = malloc(     K * sizeof(        float) );
        complex float *block = malloc( M * N * sizeof(complex float) );

        // for every block...
        #pragma omp for collapse(4)
        for (int c  = 0;  c < C;    c++) {
        for (int x = -sx; x < X; x += b) {
        for (int y = -sy; y < Y; y += b) {
        for (int z = -sz; z < Z; z += b) {

            // block[:] = 0
            memset(block, 0, M * N * sizeof(complex float));

            // load block
            for (int t0 = 0; t0 < T0; t0++) {
            for (int t1 = 0; t1 < T1; t1++) {
            for (int bx = 0; bx < b; bx++) { if (0 <= x+bx && x+bx < X) {
            for (int by = 0; by < b; by++) { if (0 <= y+by && y+by < Y) {
            for (int bz = 0; bz < b; bz++) { if (0 <= z+bz && z+bz < Z) {
                  imgs[ IDX_IMG(t0,c,t1,x+bx,y+by,z+bz) ] = block[ IDX_BLK(t0,t1,bx,by,bz) ];
            }}}}}}}}

            // U, s, V = svd(block)
            LAPACKE_cgesvd( LAPACK_COL_MAJOR, 'S', 'S',
                M, N, block, ldc, s, U, ldu, V, ldv, superb );

            // sV = thresh(s) * V
            for (int k = 0; k < K; k++)
                for (int n = 0; n < N; n++)
                    V[k*N+n] *= max(s[k]-thresh, 0);

            // block = U * sV
            cblas_cgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                 M, N, K, &alpha, U, ldu, V, ldv, &beta, block, ldc );

            // restore block
            for (int t0 = 0; t0 < T0; t0++) {
            for (int t1 = 0; t1 < T1; t1++) {
            for (int bx = 0; bx < b; bx++) { if (0 <= x+bx && x+bx < X) {
            for (int by = 0; by < b; by++) { if (0 <= y+by && y+by < Y) {
            for (int bz = 0; bz < b; bz++) { if (0 <= z+bz && z+bz < Z) {
                  block[ IDX_BLK(t0,t1,bx,by,bz) ] = imgs[ IDX_IMG(t0,c,t1,x+bx,y+by,z+bz) ];
            }}}}}}}}

        }}}}

        free(block); free(U); free(V); free(s); free(superb);
    }
}
