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

int main() {
    int limit = 1000, N = 1000;
    int *H = (int *)malloc(sizeof(int)*N*N);
    int *G = (int *)malloc(sizeof(int)*N*N);
    initialize(H, N);
    
    #pragma omp parallel 
    {  
        for (int iteration = 0; iteration < limit; iteration++) {
            #pragma omp for collapse(2)
            for (int i = 1; i < N-1; i++)
                for (int j = 1; j < N-1; j++)
                    G[N*i+j] = 0.25*(H[N*i+j-N] + H[N*i+j-1] + H[N*i+j+1] + H[N*i+j+N]);

            #pragma omp for collapse(2)
            for (int i = 1; i < N-1; i++)
                for (int j = 1; j < N-1; j++)
                    H[N*i+j] = G[N*i+j];
        }
    }

    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++)
    //         cout << H[N*i+j] << " ";
    //     cout << endl;
    // }
    return 0;
}