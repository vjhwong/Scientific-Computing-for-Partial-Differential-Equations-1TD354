import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as splg
import matplotlib.pyplot as plt
from rungekutta4 import step
from cgs import conjugate_gradient_solve


def mass_matrix_assembler(x):
    """
    Assembles mass matrix
    Input is vector x of node coordinates
    """
    N = len(x) - 1
    h = x[1] - x[0]  # Assume equidistant nodes
    A = (h / 6) * (
        1 * np.diag(np.ones(N - 1), 1)
        + 1 * np.diag(np.ones(N - 1), -1)
        + 4 * np.diag(np.ones(N), 0)
    )
    A = spsp.csr_matrix(A)
    return A.tocsr()


def stiffness_matrix_assembler(x):
    """
    Assembles stiffness matrix
    Input is vector x of node coordinates
    """
    N = len(x) - 1
    h = x[1] - x[0]  # Assume equidistant nodes
    A = (1 / h) * (
        1 * np.diag(np.ones(N - 1), 1)
        + 1 * np.diag(np.ones(N - 1), -1)
        + 2 * np.diag(np.ones(N), 0)
    )
    A = spsp.csr_matrix(A)
    return A.tocsr()


def load_vector_assembler(x):
    """
    Returns the assembled load vector b.
    Input is a vector x of node coords.
    """
    N = len(x) - 1
    B = np.zeros(N + 1)
    h = x[1] - x[0]
    B = h * f(x)
    return B


def f(x, k=100 * 2 * np.pi):
    return np.sin(k * x)


def u_exact(x, t, a, k):
    return np.sin(k * x) * np.exp(-a * t * k**2)


def run_simulation(n, method):
    """
    Run simulation of heat equation in 1D with
    homogeneous Dirichlet boundary conditions.

    Inputs:
    n: number of intervals
    method: Which solver to use LU or CGS

    Outputs:

    """

    # model parameters:
    T = 10e-5  # End time
    a = 1  # Heat equation coefficient
    xl = 0  # Left boundary
    xr = 1  # Right boundary
    L = xr - xl  # Domain length

    # Space discretization
    h = L / n
    xvec = np.linspace(xl, xr, n + 1)

    # Time discretization
    ht_try = 0.1 * np.sqrt(a) * h**2
    mt = int(np.ceil(T / ht_try) + 1)  # round up so that (mt-1)*ht = T
    tvec, ht = np.linspace(0, T, mt, retstep=True)

    # Assemble stiffness, mass matrix and load vector
    A = stiffness_matrix_assembler(xvec)
    M = mass_matrix_assembler(xvec)
    B = load_vector_assembler(xvec)

    # Initialize time and solution
    t = 0
    if method == "LU":
        lu = splg.splu(A)
        u = lu.solve(B)
    elif method == "CGS":
        tol = 1e-6
        u, n_iter = conjugate_gradient_solve(A, B, tol=tol)
    else:
        raise NotImplementedError("Method not implemented")


def main():
    n_vec = [2_000, 4_000, 8_000]
    pass


if __name__ == "__main__":
    main()
