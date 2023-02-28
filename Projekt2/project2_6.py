import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as splg
import matplotlib.pyplot as plt
from rungekutta4 import step


def stiffness_matrix_assembler(x):
    """
    Assembles stiffness matrix
    Inpus is vector x of node coordinates
    """
    N = len(x) - 1  # number of elements
    A = spsp.dok_matrix((N + 1, N + 1))  # initialize stiffness matrix
    for i in range(N):  # loop over elements
        h = x[i + 1] - x[i]  # element length
        A[i, i] += 1 / h  # assemble element stiffness
        A[i, i + 1] += -1 / h
        A[i + 1, i] += -1 / h
        A[i + 1, i + 1] += 1 / h
    A[0, 0] = 1e6  # adjust for BC
    A[N, N] = 1e6
    return A.tocsr()


def my_load_vector_assembler(x):
    """
    Returns the assembled load vector b.
    Input is a vector x of node coords.
    """
    N = len(x) - 1
    B = np.zeros(N + 1)
    for i in range(N):
        h = x[i + 1] - x[i]
        B[i] = B[i] + f(x[i]) * h / 2
        B[i + 1] = B[i + 1] + f(x[i + 1]) * h / 2
    return B


def f(x, k=100 * 2 * np.pi):
    return np.sin(k * x)


def main():
    pass


if __name__ == "__main__":
    main()
