import numpy as np
import numpy.linalg as nplg
from system_matrix import system_matrix


class MatrixNotSquareError(ValueError):
    pass


def jacobi_setup(A):
    """Splits the matrix A and returns appropriate matrices"""

    # L: strict lower triangular part
    # D: diagonal part
    # U: strict upper triangular part

    if A.shape[0] != A.shape[1]:
        raise MatrixNotSquareError("Input matrix is not square")
    L = np.tril(A, k=-1)
    D = np.diag(np.diag(A))
    U = np.triu(A, k=1)
    return L, D, U


def jacobi_solve(Dinv, L_plus_U, b, x0=None, tol=1e-6):
    """Solves a linear system Ax=b using the Jacobi method"""

    # Dinv:     inv(D), where D is the diagonal part of A
    # L_plus_U: L+U (upper and lower triangular parts of A)
    # b:        right-hand side vector
    # x0:       Initial guess (if None, the zero vector is used)
    # tol:      Relative error tolerance

    # If no initial guess supplied, use the zero vector
    if x0 is None:
        x_new = 0 * b
    else:
        x_new = x0

    # Initialize number of iterations and error
    n_iter = 0
    err = 2 * tol

    # Iterate until tol is reached
    while err > tol:
        # Add counter
        n_iter += 1

        # Store the previous solution in x
        x = x_new

        # Jacobi method
        x_new = -Dinv @ L_plus_U @ x + Dinv @ b

        # Caclulate the relative error
        err = nplg.norm(x_new - x) / nplg.norm(x)

    # Return approximate solution and number of iterations required
    return x, n_iter


def main():
    pass


if __name__ == "__main__":
    main()
