#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

void initialize(int *A, int N) {
    for (int i = 0; i < N; i++)
        A[i] = (rand()%10 + 1);
}


__global__ void get_stats(int *A, int *min, int *max, int *sum, int *square_sum, int N) {
    int index = threadIdx.x;
    if (index < N) {
        atomicMin(min, A[index]);
        atomicMax(max, A[index]);
        atomicAdd(sum, A[index]);
        atomicAdd(square_sum, A[index]*A[index]);
    }
}

__global__ void get_sum(int *A, int *sum, int N) {
    int index = threadIdx.x;
    if (index < N)
        atomicAdd(sum, A[index]);
}

__global__ void get_square_sum(int *A, int *square_sum, int N) {
    int index = threadIdx.x;
    if (index < N)
        atomicAdd(square_sum, A[index]*A[index]);
}

__global__ void get_min(int *A, int *min, int N) {
    int index = threadIdx.x;
    if (index < N)
        atomicMin(min, A[index]);
}

__global__ void get_max(int *A, int *max, int N) {
    int index = threadIdx.x;
    if (index < N)
        atomicMax(max, A[index]);
}

int main() {
    int N = 1e+3;
    int min = INT_MAX, max = INT_MIN, sum = 0, square_sum = 0;
    int *d_min, *d_max, *d_sum, *d_square_sum;

    cudaMalloc((void **)&d_min, sizeof(int));
    cudaMalloc((void **)&d_max, sizeof(int));
    cudaMalloc((void **)&d_sum, sizeof(int));
    cudaMalloc((void **)&d_square_sum, sizeof(int));

    int *d_A;
    int *A = (int *) malloc(sizeof(int)*N);
    initialize(A, N);

    cudaMalloc((void **)&d_A, sizeof(int)*N);
    cudaMemcpy(d_A, A, sizeof(int)*N, cudaMemcpyHostToDevice);

    cudaMemcpy(d_min, &min, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &max, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &sum, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_square_sum, &square_sum, sizeof(int), cudaMemcpyHostToDevice);
    get_stats<<<1, N>>>(d_A, d_min, d_max, d_sum, d_square_sum, N);
    cudaMemcpy(&min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&square_sum, d_square_sum, sizeof(int), cudaMemcpyDeviceToHost);

    float mean = sum/(N*1.0);
    float SD = sqrt((square_sum+2*mean*sum+N*mean*mean)/(N*1.0));
    cout << "Min: " << min << endl;
    cout << "Max: " << max << endl;
    cout << "Mean: " << mean << endl;
    cout << "SD: " << SD << endl;
    return 0;
}