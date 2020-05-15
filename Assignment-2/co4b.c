#include <stdio.h>
#include <stdlib.h>
#define n 1000

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

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double t = B[i][j];
            B[i][j] = B[j][i];
            B[j][i] = t;
        }
    }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k]*B[j][k];

    return 0;
}