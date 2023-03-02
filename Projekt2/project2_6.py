import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as splg
import matplotlib.pyplot as plt
import time
from rungekutta4 import FEM_step
from cgs import conjugate_gradient_solve


def mass_matrix_assembler(x):
    """
    Assembles mass matrix
    Input is vector x of node coordinates
    """
    N = len(x)
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
    N = len(x)
    h = x[1] - x[0]  # Assume equidistant nodes
    A = (1 / h) * (
        1 * np.diag(np.ones(N - 1), 1)
        + 1 * np.diag(np.ones(N - 1), -1)
        + (-2) * np.diag(np.ones(N), 0)
    )
    A = spsp.csr_matrix(A)
    return A.tocsr()


def f(x, k=100 * 2 * np.pi):
    """Calculates initial data"""
    return np.sin(k * x)


def run_simulation(n: int, method: str):
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
    xvec = np.linspace(xl + h, xr - h, n - 1)

    # Time discretization
    ht_try = 0.1 * np.sqrt(a) * h**2
    mt = int(np.ceil(T / ht_try) + 1)  # round up so that (mt-1)*ht = T
    _, ht = np.linspace(0, T, mt, retstep=True)  # ht = size of timestep

    # Assemble stiffness and mass matrix
    A = stiffness_matrix_assembler(xvec)
    M = mass_matrix_assembler(xvec)

    def rhs(A, u, M, method):
        tol = 1e-6
        A_u = A @ u
        if method == "CGS":
            u, _ = conjugate_gradient_solve(M, A_u, tol=tol)
            return u
        elif method == "LU":
            lu = splg.splu(M)
            return lu.solve(A_u)
        else:
            raise NotImplementedError("Method is not implemented")

    # Initialize time and solution vector
    t = 0
    u = f(xvec)

    # Measure time it takes to run
    time_start = time.perf_counter()
    for _ in range(mt - 1):
        u, t = FEM_step(rhs, A, u, M, method, t, ht)
    time_end = time.perf_counter()

    print(f"Method: {method}\nn: {n}\nRuntime: {(time_end - time_start)}")
    # Return solution vector and runtime
    return u, (time_end - time_start)


def calculate_u_exact(n, t=10e-5, a=1, k=100 * 2 * np.pi, xl=0, xr=1):
    """Calculate the exact solution"""
    h = (xr - xl) / n
    x = np.linspace(xl + h, xr - h, n - 1)
    return np.sin(k * x) * np.exp(-a * t * k**2)


def relative_error(u, u_exact):
    """Calculate the relative error"""
    print(np.linalg.norm(u - u_exact) / np.linalg.norm(u_exact))
    return np.linalg.norm(u - u_exact) / np.linalg.norm(u_exact)


def main():
    # Lists that contain n and methods we want to evaluate
    n_vec = [2_000, 4_000, 8_000]
    methods = ["CGS", "LU"]

    # Initialize arrays to hold the relative errors and runtime
    rel_error = np.zeros((len(methods), len(n_vec)))
    runtime = np.zeros((len(methods), len(n_vec)))

    # Calculate the exact solution for all n
    u_exacts = [calculate_u_exact(n) for n in n_vec]

    # Run the simulation for all methods using each n
    # Add the runtime and relative error to arrays
    for i, method in enumerate(methods):
        for j, (n, u_exact) in enumerate(zip(n_vec, u_exacts)):
            u, time = run_simulation(n, method)
            runtime[i][j] = time
            rel_error[i][j] = relative_error(u, u_exact)
    print(runtime)
    print()
    print(rel_error)


if __name__ == "__main__":
    main()
