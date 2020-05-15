#include <stdio.h>
#include <cuda_runtime.h>

void initialize(int *H, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            H[N*i+j] = 0;

    for (int i = 0; i < N; i++) {
        H[N*i] = H[N*i+N-1] = H[N*(N-1)+i] = 20;
        H[i] = i >= ((N*30)/100) && i < ((N*70)/100) ? 100 : 20;
    }
}

void serial_heat_distribution(int *H, int *G, int N, int limit) {
    for (int iteration = 0; iteration < limit; iteration++) {
        for (int i = 1; i < N-1; i++)
            for (int j = 1; j < N-1; j++)
                G[N*i+j] = 0.25*(H[N*i+j-N] + H[N*i+j-1] + H[N*i+j+1] + H[N*i+j+N]);

        for (int i = 1; i < N-1; i++)
            for (int j = 1; j < N-1; j++)
                H[N*i+j] = G[N*i+j];
    }
}

__global__ void heat_distribution(int *H, int *G, int N, int limit) {
    for (int iteration = 0; iteration < limit; iteration++) {
        if (blockIdx.x > 0 && threadIdx.x > 0 && blockIdx.x < N-1 && threadIdx.x < N-1) {
            int index = N*blockIdx.x + threadIdx.x;
            G[index] = 0.25*(H[index-N] + H[index-1] + H[index+1] + H[index+N]);
            __syncthreads();
            H[index] = G[index];
            __syncthreads();
        }
    }
}

int main(void) {

    int N = 1000, limit = 1000;
    int *H, *d_H, *G, *d_G;

    cudaMalloc((void **)&d_H, sizeof(int)*N*N);
    cudaMalloc((void **)&d_G, sizeof(int)*N*N);
    
    H = (int *)malloc(sizeof(int)*N*N);
    G = (int *)malloc(sizeof(int)*N*N); 
    initialize(H, N);

    cudaMemcpy(d_H, H, sizeof(int)*N*N, cudaMemcpyHostToDevice);
    heat_distribution<<<N, N>>>(d_H, d_G, N, limit);
    cudaMemcpy(H, d_H, sizeof(int)*N*N, cudaMemcpyDeviceToHost);

    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++)
    //         printf("%d ", H[N*i+j]);
    //     printf("\n");
    // }

    free(H); free(G);
    cudaFree(d_H); cudaFree(d_G);
    return 0;
}