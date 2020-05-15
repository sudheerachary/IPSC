#include <stdio.h>
#include <stdlib.h>
#define n 1000
#define SM (CLS/sizeof(double))
#define min(x, y) (((x) < (y)) ? (x) : (y))

int main()
{
    double *A[n], *B[n], *C[n];
    for (int i = 0; i < n; i++) {
        A[i] = (double *) malloc(n * sizeof(double));
        B[i] = (double *) malloc(n * sizeof(double));
        C[i] = (double *) malloc(n * sizeof(double));
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = 1;
            B[i][j] = 1;
            C[i][j] = 1;
        }
    }

    for (int k = 0; k < n; k += SM)
        for (int j = 0; j < n; j += n)
            for (int i = 0; i < n; ++i)
                for (int jj = j; jj < min(j + SM, n); ++jj)
                    for (int kk = k; kk < min(k + SM, n); ++kk)
                        C[i][jj] += A[i][kk] * B[kk][jj];
    return 0;
}