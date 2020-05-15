#include <omp.h>
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

void bitonic_sort(int start, int end, int *A) {
    int length = end-start+1;
    for (int j = 2; j <= length; j = j*2 ) {
        #pragma omp parallel for
        for (int i = 0; i < length; i = i+j) {
            if ((i/j)%2 == 0)
                merge(i, i+j-1, 1, A);
            else
                merge(i, i+j-1, 0, A);
        }
    }
}

int main() {
    int N = 6;
    int *A = (int *)malloc(sizeof(int)*(1 << N));
    initialize(A, (1 << N));

    bitonic_sort(0, (1 << N)-1, A);
    for (int i = 0; i < (1 << N); i++)
        cout << A[i] << " ";
    cout << endl;
}