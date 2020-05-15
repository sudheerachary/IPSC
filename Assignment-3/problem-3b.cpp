#include <mpi.h>
#include <bits/stdc++.h>
using namespace std;

void initialize(int *A, int N) {
    for (int i = 0; i < N; i++)
        A[i] = (rand()%N + 1);
}

void swap(int *A, int i, int j) {
    int temp = A[i];
    A[i] = A[j];
    A[j] = temp;
}

void merge(int start, int end, int dir, int *A) {
    int length = end-start+1;
    for (int j = length/2; j > 0; j = j/2) {
        for (int i = start; i+j < start+length; i++)
            if (dir == (A[i] > A[i+j]))
                swap(A, i, i+j);
    }
}

void bitonic_sort(int start, int end, int *A, int rank, int size) {
    MPI_Status status;
    int length = end-start+1;
    int *B = (int *)malloc(sizeof(int)*length);
    for (int j = 2; j <= length; j = j*2 ) {
        if (rank == 0) {
            for (int k = 1; k < size; k++) {
                MPI_Recv(B, length, MPI_INT, k, 1, MPI_COMM_WORLD, &status);
                if ((k-1)*j < length) {
                    for (int l = (k-1)*j; l < k*j; l++)
                        A[l] = B[l];
                }
            }
            for (int k = 1; k < size; k++)
                MPI_Send(A, length, MPI_INT, k, 1, MPI_COMM_WORLD);
        }
        else {
            int i = (rank-1)*j;
            if (i < length) {
                if ((i/j)%2 == 0)
                    merge(i, i+j-1, 1, A);
                else            
                    merge(i, i+j-1, 0, A);
            }
            MPI_Send(A, length, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Recv(A, length, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        }
    }
}

int main(int argc, char *argv[]) {
    int N = 6, rank, size;
    int *A = (int *)malloc(sizeof(int)*(1 << N));
    initialize(A, (1 << N));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    bitonic_sort(0, (1 << N)-1, A, rank, size);

    if (rank == 0) {
        for (int i = 0; i < (1 << N); i++)
            cout << A[i] << " ";
        cout << endl;
    }

    MPI_Finalize();
    return 0;
}