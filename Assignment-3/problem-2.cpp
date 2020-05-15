#include <omp.h>
#include <bits/stdc++.h>

using namespace std;

int main() {
    int N = 100, brr[N][N][N][N], arr[N][N][N][N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                for (int l = 0; l < N; l++) {
                    arr[i][j][k][l] = 0;
                    brr[i][j][k][l] = 0;
                }
            }
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 10; k < N; k++) {
                for (int l = 10; l < N; l++) {
                    brr[i][j][k][l] = i + j*2 + brr[i][j][k-10][l];
                }
            }
        }
    }
    
    /*
        - k-th loop cannot be run parallely as it is
          dependent on previous cells, so change in order
          of loops can be run completely parallel.
    */
    #pragma omp parallel 
    {
        #pragma omp for collapse(3)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int l = 10; l < N; l++) {
                    for (int k = 10; k < N; k++) {
                        arr[i][j][k][l] = i + j*2 + arr[i][j][k-10][l];
                    }
                }
            }
        }
    }

    bool flag = true;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                for (int l = 0; l < N; l++) {
                    if (arr[i][j][k][l] != brr[i][j][k][l]) {
                        flag = false;
                        break;
                    }
                }
            }
        }
    }
    cout << "Result: " << flag << endl;
    
    return 0;
}