#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

void initialize(int *A, int N) {
    for (int i = 0; i < N; i++)
        A[i] = (rand()%N + 1);
}

__device__ void swap(int *A, int i, int j) {
    int temp = A[i];
    A[i] = A[j];
    A[j] = temp;
}

__device__ void cuda_merge(int start, int end, int dir, int *A) {
    int length = end-start+1;
    for (int j = length/2; j > 0; j = j/2) {
        for (int i = start; i+j < start+length; i++)
            if (dir == (A[i] > A[i+j]))
                swap(A, i, i+j);
    }
}

__global__ void cuda_sort(int *A, int length) {    
    int start = blockIdx.x*length;
    if ((start/length)%2 == 0)
        cuda_merge(start, start+length-1, 1, A);
    else
        cuda_merge(start, start+length-1, 0, A);
}

void bitonic_sort(int start, int end, int *A) {
    int length = end-start+1;
    for (int j = 2; j <= length; j = j*2)
        cuda_sort<<<length/j, 1>>>(A, j);
}

int main(void) {

    int N = 6, *A, *d_A;
    A = (int *)malloc(sizeof(int)*(1 << N));
    cudaMalloc((void **)&d_A, sizeof(int)*(1 << N));
    initialize(A, (1 << N));

    cudaMemcpy(d_A, A, sizeof(int)*(1 << N), cudaMemcpyHostToDevice);
    bitonic_sort(0, (1 << N)-1, d_A);
    cudaMemcpy(A, d_A, sizeof(int)*(1 << N), cudaMemcpyDeviceToHost);

    for (int i = 0; i < (1 << N); i++)
        printf("%d ", A[i]);
    printf("\n");

    free(A); cudaFree(d_A);
    return 0;
}