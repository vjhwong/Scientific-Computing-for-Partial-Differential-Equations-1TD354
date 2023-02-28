import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as splg
import matplotlib.pyplot as plt


def stiffness_matrix_assembler(x):
    """
    Assembles stiffness matrix given a vector x of node
    coordinates
    """
    N = len(x) - 1  # number of elements
    A = spsp.dok_matrix((N + 1, N + 1))  # initialize stiffness matrix
    for i in range(N):  # loop over elements
        h = x[i + 1] - x[i]  # element length
        # A[i, i] += 1 / h  # assemble element stiffness
        # A[i, i + 1] += -1 / h
        # A[i + 1, i] += -1 / h
        # A[i + 1, i + 1] += 1 / h
        A[i, i] += 1  # assemble element stiffness
        A[i, i + 1] += -1
        A[i + 1, i] += -1
        A[i + 1, i + 1] += 1
    A[0, 0] = 1e6  # adjust for BC
    A[N, N] = 1e6
    return A.tocsr()


def main():
    x = stiffness_matrix_assembler(np.linspace(0, 1, 6))
    print(x.toarray())


if __name__ == "__main__":
    main()
