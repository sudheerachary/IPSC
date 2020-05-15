#include <omp.h>
#include <bits/stdc++.h>
using namespace std;

void initialize(float **A, float *b, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = rand()%20+1;

    for (int i = 0; i < 3; i++)
        b[i] = rand()%20+1;
}

class Vector {
    public:
        int _rows;
        float *_vector;

    Vector(int rows=0, float *vector=NULL)  {
        _rows = rows;
        _vector = (float *) malloc(sizeof(float)*_rows);
        if (vector == NULL) {
            #pragma omp parallel for
            for (int i = 0; i < _rows; i++)
                _vector[i] = 0.0;
        }
        else {
            #pragma omp parallel for
            for (int i = 0; i < _rows; i++)
                _vector[i] = vector[i];
        }
    }
};

class Matrix {
    public:
        int _rows, _cols;
        float **_matrix;

    Matrix(int rows=0, int cols=0, float **matrix=NULL) {
        _rows = rows;
        _cols = cols;
        _matrix = (float **) malloc(sizeof(float *)*_rows);

        #pragma omp parallel for
        for (int i = 0; i < _rows; i++)
            _matrix[i] = (float *) malloc(sizeof(float *)*_cols);

        if (matrix == NULL) {
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < _rows; i++) {
                for (int j = 0; j < _cols; j++) {
                    if (i == j)
                        _matrix[i][j] = 1.0;
                    else
                        _matrix[i][j] = 0.0;
                }
            }
        }
        else {            
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < _rows; i++)
                for (int j = 0; j < _cols; j++)
                    _matrix[i][j] = matrix[i][j];
        }   
    }
};

class LUdecompose : public Matrix {
    public:
        Matrix lower, upper, triangular;
        LUdecompose(int rows, int cols, float **matrix): Matrix {rows, cols, matrix} {}
        void generate_triangular_matrix();
        Vector forward_substitution(Vector );
        Vector backward_substitution(Vector );
        Vector lu_solve(Vector );
        float error(Vector, Vector);
};

void LUdecompose::generate_triangular_matrix() {
    lower = Matrix(_rows, _cols);
    upper = Matrix(_rows, _cols);
    triangular = Matrix(_rows, _cols, _matrix);
    
    for (int j = 0; j < _cols; j++) {
        #pragma omp parallel for
        for (int i = j+1; i < _rows; i++)
            triangular._matrix[i][j] /= triangular._matrix[j][j];

        #pragma omp parallel for collapse(2)
        for (int i = j+1; i < _rows; i++)
            for (int k = j+1; k < _cols; k++)
                triangular._matrix[i][k] -= triangular._matrix[i][j]*triangular._matrix[j][k];
    }

    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < _rows; i++)
        for (int j = i; j < _cols; j++)
            upper._matrix[i][j] = triangular._matrix[i][j];

    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < _rows; i++)
        for (int j = 0; j < i; j++)
            lower._matrix[i][j] = triangular._matrix[i][j];
}

Vector LUdecompose::forward_substitution(Vector b) {
    Vector t = Vector(b._rows);
    for (int i = 0; i < b._rows; i++) {
        float carry = 0;
        #pragma omp parallel for
        for (int j = 0; j < i; j++) {
            #pragma omp atomic
            carry += lower._matrix[i][j]*t._vector[j];
        }
        t._vector[i] = (b._vector[i]-carry)/lower._matrix[i][i];
    }
    return t;
}

Vector LUdecompose::backward_substitution(Vector b) {
    Vector x = Vector(b._rows);
    for (int i = b._rows-1; i > -1; i--) {
        float carry = 0;
        #pragma omp parallel for
        for (int j = i+1; j < b._rows; j++) {
            #pragma omp atomic
            carry += upper._matrix[i][j]*x._vector[j];
        }
        x._vector[i] = (b._vector[i]-carry)/upper._matrix[i][i];
    }
    return x;
}

Vector LUdecompose::lu_solve(Vector b) {
    generate_triangular_matrix();
    Vector t = forward_substitution(b);
    Vector x = backward_substitution(t);
    cout << "Error: " << error(x, b) << endl;
    return x;
}

float LUdecompose::error(Vector x, Vector b) {
    // mean squared error on || b-Ax ||
    float _error = 0;
    Vector y = Vector(b._rows);
    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _cols; j++) {
            y._vector[i] += _matrix[i][j]*x._vector[j];
        }
        _error += (y._vector[i]-b._vector[i])*(y._vector[i]-b._vector[i]);
    }
    return sqrt(_error);
}

int main() {

    int N = 10;
    float *b = (float *) malloc(sizeof(float)*N);
    float **A = (float **) malloc(sizeof(float *)*N);
    for (int i = 0; i < N; i++)
        A[i] = (float *) malloc(sizeof(float *)*N);
    initialize(A, b, N);

    LUdecompose D = LUdecompose(N, N, A);
    Vector x = D.lu_solve(Vector(N, b));

    for (int i = 0; i < N; i++)
        cout << x._vector[i] << " ";
    cout << endl;
    return 0;
}