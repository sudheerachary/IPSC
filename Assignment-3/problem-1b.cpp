#include <mpi.h>
#include <bits/stdc++.h>

using namespace std;

void initialize(int *H, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            H[N*i+j] = 0;

    for (int i = 0; i < N; i++) {
        H[N*i] = H[N*i+N-1] = H[N*(N-1)+i] = 20;
        H[i] = i >= ((N*30)/100) && i < ((N*70)/100) ? 100 : 20;
    }
}

void heat_distribution(int *H, int *G, int rank, int size, int N, int limit) {
    MPI_Status status;
    int work = N/(size-1), start = (rank-1)*work;
    for (int iteration = 0; iteration < limit; iteration++) {
        for (int i = max(1, start); i < min(N-1, start+work); i++)
            for (int j = 1; j < N-1; j++)
                G[N*i+j] = 0.25*(H[N*i+j-N] + H[N*i+j-1] + H[N*i+j+1] + H[N*i+j+N]);

        for (int i = max(1, start); i < min(N-1, start+work); i++)
            for (int j = 1; j < N-1; j++)
                H[N*i+j] = G[N*i+j];

        MPI_Send(H, N*N, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Recv(H, N*N, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
    }
}

int main(int argc, char *argv[]) {

    int rank, size, limit = 1000, N = 1000;
    int *H = (int *)malloc(sizeof(int)*N*N);
    int *G = (int *)malloc(sizeof(int)*N*N);
    int *F = (int *)malloc(sizeof(int)*N*N);

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    initialize(H, N);
    
    if (rank == 0) {
        for (int iteration = 0; iteration < limit; iteration++) {
            for (int k = 1; k < size; k++) {
                MPI_Recv(F, N*N, MPI_INT, k, 1, MPI_COMM_WORLD, &status);
                int work = N/(size-1), start = (k-1)*work;
                for (int i = max(1, start); i < min(N-1, start+work); i++)
                    for (int j = 1; j < N-1; j++)
                        H[N*i+j] = F[N*i+j];
            }
            for (int k = 1; k < size; k++)
                MPI_Send(H, N*N, MPI_INT, k, 1, MPI_COMM_WORLD);
        }
        // for (int i = 0; i < N; i++) {
        //     for (int j = 0; j < N; j++)
        //         printf("%d ", H[N*i+j]);
        //     printf("\n");
        // }
    }
    else
        heat_distribution(H, G, rank, size, N, limit);

    MPI_Finalize();
    return 0;
}