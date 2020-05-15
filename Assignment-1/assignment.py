import math
from random import randint

class Vector(object):
    def __init__(self, rows, vector=None):
        self.rows = rows
        self.vector = [0]*self.rows
        if not vector is None:
            for i in range(self.rows):
                self.vector[i] = vector[i]


class Matrix(object):
    def __init__(self, rows, cols, matrix=None):
        self.rows = rows
        self.cols = cols
        self.matrix = [[0]*self.cols for i in range(self.rows)]
        if not matrix is None:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.matrix[i][j] = matrix[i][j]

    def generate_random_matrix(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] = randint(1, 100)

    def generate_identity(self):
        self.matrix = [[0]*i + [1] + [0]*(self.cols-i-1) for i in range(self.rows)]


class BandedMatrix(Matrix):
    def __init__(self, rows, cols, matrix=None, lower_bandwidth=0, upper_bandwidth=0):
        super(BandedMatrix, self).__init__(rows, cols, matrix=matrix)
        self.lower_bandwidth = lower_bandwidth
        self.upper_bandwidth = upper_bandwidth
        self.banded_matrix = [[0]*max(self.rows, self.cols) for i in range(self.lower_bandwidth+self.upper_bandwidth+1)]

    def generate_banded_matrix(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if not i > j+self.lower_bandwidth and not j > i+self.upper_bandwidth:
                    self.matrix[i][j] = randint(1, 100)

    def store_banded_matrix(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if i+self.upper_bandwidth-j >= 0 and i+self.upper_bandwidth-j < self.lower_bandwidth+self.upper_bandwidth+1:
                    self.banded_matrix[i+self.upper_bandwidth-j][j] = self.matrix[i][j]


class SparseMatrix(Matrix):
    def __init__(self, rows, cols, matrix=None, fill=0.1):
        super(SparseMatrix, self).__init__(rows, cols, matrix=matrix)
        self.fill = self.rows*self.cols*fill

    def generate_sparse_matrix(self):
        count = 0
        while count < self.fill:
            i = randint(0, self.rows-1)
            j = randint(0, self.cols-1)
            if self.matrix[i][j] == 0:
                self.matrix[i][j] = randint(1, 100)
                count += 1

    def store_sparse_matrix(self, form="coo"):
        self.val = []
        self.col_indices = []
        if form == "coo":
            self.row_indices = []
            for i in range(self.rows):
                for j in range(self.cols):
                    if not self.matrix[i][j] == 0:
                        self.row_indices.append(i)
                        self.col_indices.append(j)
                        self.val.append(self.matrix[i][j])
        elif form == "csr":
            _row_pointer = 0
            self.row_pointers = [0]
            for i in range(self.rows):
                for j in range(self.cols):
                    if not self.matrix[i][j] == 0:
                        _row_pointer += 1
                        self.col_indices.append(j)
                        self.val.append(self.matrix[i][j])
                self.row_pointers.append(_row_pointer)
        else:
            print("Unknown format: {}".format(form))


class LUdecompose(Matrix):
    def __init__(self, rows, cols, matrix=None):
        super(LUdecompose, self).__init__(rows, cols, matrix=matrix)

    def gauss_transform(self, A, j):
        M = Matrix(self.rows, self.cols)
        M.generate_identity()
        for i in range(j+1, self.rows):
            M.matrix[i][j] -= A.matrix[i][j]/A.matrix[j][j]
        return M

    def mylu_generate_triangular_matrix(self):
        self.lower_triangular_matrix = Matrix(self.rows, self.cols)
        self.upper_triangular_matrix = Matrix(self.rows, self.cols)
        self.triangular_matrix = Matrix(self.rows, self.cols, matrix=self.matrix)
        for j in range(self.cols):
            for i in range(j+1, self.rows):
                self.triangular_matrix.matrix[i][j] /= self.triangular_matrix.matrix[j][j]
            for i in range(j+1, self.rows):
                for k in range(j+1, self.cols):
                    self.triangular_matrix.matrix[i][k] -= self.triangular_matrix.matrix[i][j]*self.triangular_matrix.matrix[j][k]

        for i in range(self.rows):
            for j in range(i, self.cols):
                self.upper_triangular_matrix.matrix[i][j] = self.triangular_matrix.matrix[i][j]
        
        self.lower_triangular_matrix.generate_identity()
        for i in range(self.rows):
            for j in range(i):
                self.lower_triangular_matrix.matrix[i][j] = self.triangular_matrix.matrix[i][j]

    def foreward_substitution(self, b):
        t = Vector(b.rows)
        for i in range(b.rows):
            carry = 0
            for j in range(i):
                carry += self.lower_triangular_matrix.matrix[i][j]*t.vector[j]
            t.vector[i] = (b.vector[i] - carry)/self.lower_triangular_matrix.matrix[i][i]
        return t

    def backward_substitution(self, b):
        x = Vector(b.rows)
        for i in range(b.rows-1, -1, -1):
            carry = 0
            for j in range(i+1, b.rows):
                carry += self.upper_triangular_matrix.matrix[i][j]*x.vector[j]
            x.vector[i] = (b.vector[i] - carry)/self.upper_triangular_matrix.matrix[i][i]
        return x

    def lu_solve(self, b):
        self.mylu_generate_triangular_matrix()
        t = self.foreward_substitution(b)
        x = self.backward_substitution(t)
        print("Error: {}".format(self.error(x, b)))
        return x

    def error(self, x, b):
        _error = 0
        y = dense_vecmul(Matrix(self.rows, self.cols, matrix=self.matrix), x)
        for i in range(y.rows):
            _error += (y.vector[i] - b.vector[i])**2
        return math.sqrt(_error)


def dense_vecmul(A, b):
    y = None
    try:
        if not A.cols == b.rows:
            print("dimension mis-match")    
            return None

        y = Vector(A.rows)
        for i in range(A.rows):
            for j in range(A.cols):
                y.vector[i] += A.matrix[i][j]*b.vector[j]
    except Exception as e:
        print(e)
    return y

def dense_matmul(A, B):
    C = None
    try:
        if not A.cols == B.rows:
            print("dimension mis-match")    
            return None

        C = Matrix(A.rows, B.cols)
        for i in range(C.rows):
            for j in range(C.cols):
                for k in range(A.cols):
                    C.matrix[i][j] += A.matrix[i][k]*B.matrix[k][j]
    except Exception as e:
        print(e)
    return C

def banded_matmul(A, B):
    C = None
    try:
        if not A.cols == B.rows:
            print("dimension mis-match")    
            return None

        C = Matrix(A.rows, B.cols)
        for i in range(C.rows):
            for j in range(C.cols):
                for k in range(A.cols):
                    if i+A.upper_bandwidth-k >= 0 and i+A.upper_bandwidth-k < A.lower_bandwidth+A.upper_bandwidth+1 and k+B.upper_bandwidth-j >= 0 and k+B.upper_bandwidth-j < B.lower_bandwidth+B.upper_bandwidth+1:
                        C.matrix[i][j] += A.banded_matrix[i+A.upper_bandwidth-k][k]*B.banded_matrix[k+B.upper_bandwidth-j][j]
    except Exception as e:
        print(e)
    return C

def sparse_vecmul(A, b, form="coo"):
    y = None
    if form == "coo":
        try:
            if not A.cols == b.rows:
                print("dimension mis-match")
                return None

            y = Vector(A.rows)
            for i in range(len(A.val)):
                y.vector[A.row_indices[i]] += A.val[i]*b.vector[A.col_indices[i]]
        except Exception as e:
            print(e)
    elif form == "csr":
        try:
            if not A.cols == b.rows:
                print("dimension mis-match")
                return None

            y = Vector(A.rows)
            for i in range(len(A.row_pointers)-1):
                for j in range(A.row_pointers[i], A.row_pointers[i+1]):
                    y.vector[i] += A.val[j]*b.vector[A.col_indices[j]]

        except Exception as e:
            print(e)
    else:
        print("Unknown format: {}".format(form))
    return y

def sparse_matmul(A, B, form="coo"):
    C = SparseMatrix(A.rows, B.cols, matrix=[[0]*B.cols for i in range(A.rows)])
    for j in range(B.cols):
        b = Vector(B.rows, vector=[B.matrix[k][j] for k in range(B.rows)])
        y = sparse_vecmul(A, b, form=form)
        for i in range(C.rows):
            C.matrix[i][j] = y.vector[i]
    return C