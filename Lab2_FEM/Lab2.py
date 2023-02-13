import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as splg
import matplotlib.pyplot as plt

######################################################################################
##                                                                                  ##
##  Lab "Introduction to Finite Element Methods", for the course                    ##
##  "Scientific computing for PDEs" at Uppsala University.                          ##
##  Based on Matlab code in the book The Finite Element Method: Theory,             ##
##  Implementation, and Applications, by Mats G. Larson and Fredrik Bengzon.        ##
##                                                                                  ##
##  Solves the 1D Poisson equation -u''= f, on domain a < x < b using P1 finite     ##
##  elements                                                                        ##
##                                                                                  ##
##  The code has been tested on the following versions:                             ##
##  - Python 3.8.16                                                                 ##
##  - Numpy 1.24.1                                                                  ##
##  - Scipy 1.10.0                                                                  ##
##  - Matplotlib 3.6.3                                                              ##
##                                                                                  ##
######################################################################################


def my_stiffness_matrix_assembler(x):
    #
    # Returns the assembled stiffness matrix A.
    # Input is a vector x of node coords.
    #
    N = len(x) - 1  # number of elements
    A = spsp.dok_matrix((N + 1, N + 1))  # initialize stiffness matrix
    for i in range(N):  # loop over elements
        h = x[i + 1] - x[i]  # element length
        A[i, i] += 1 / h  # assemble element stiffness
        A[i, i + 1] += -1 / h
        A[i + 1, i] += -1 / h
        A[i + 1, i + 1] += 1 / h
    A[0, 0] = 1.0  # adjust for BC
    A[N, N] = 1.0
    return A.tocsr()


def my_load_vector_assembler(x):
    #
    # Returns the assembled load vector b.
    # Input is a vector x of node coords.
    #
    N = len(x) - 1
    B = np.zeros(N + 1)
    for i in range(N):
        h = x[i + 1] - x[i]
        B[i] = B[i] + f(x[i]) * h / 2
        B[i + 1] = B[i + 1] + f(x[i + 1]) * h / 2
    return B


def f(x):
    return 2


def u_exact(x):
    return x * (1 - x)


def main():

    a = 0  # left end point of interval
    b = 1  # right
    N = 100  # number of intervals
    h = (b - a) / N  # mesh size
    x = np.arange(a, b, h)  # node coords
    u = u_exact(x)
    A = my_stiffness_matrix_assembler(x)
    B = my_load_vector_assembler(x)
    xi = splg.spsolve(A, B)  # solve system of equations
    error = np.sqrt((u.T @ u - xi.T @ xi) / N)
    print(error)
    plt.plot(x, xi)  # plot solution
    plt.xlabel("x")
    plt.show()


if __name__ == "__main__":
    main()
