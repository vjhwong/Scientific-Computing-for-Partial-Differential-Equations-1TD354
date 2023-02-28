import time
import numpy as np
import scipy.sparse.linalg as spsplg
from system_matrix import system_matrix as test_matrix
from jacobi import jacobi_setup, jacobi_solve
from cgs import conjugate_gradient_solve

np.random.seed(1)


def run_and_time(n=10000, N=100, methods=["gauss", "lu", "jacobi"], tol=1e-6):
    """Solves N nxn systems using the listed methods and prints the execution time"""
    A = test_matrix(n)
    B = [np.random.rand(n) for _ in range(N)]
    iterative_methods = ["jacobi", "cg"]

    for method in methods:

        if method in iterative_methods:
            n_iter_tot = 0
        else:
            n_iter_tot = "N/A"

        if method == "lu":
            lu = spsplg.splu(A)
        elif method == "jacobi":
            Dinv, L_plus_U = jacobi_setup(A)

        start_time = time.time()
        for b in B:
            if method == "gauss":
                x = spsplg.spsolve(A, b)
            if method == "lu":
                x = lu.solve(b)
            elif method == "jacobi":
                x, n_iter = jacobi_solve(Dinv, L_plus_U, b, tol=tol)
            elif method == "cg":
                x, n_iter = conjugate_gradient_solve(A, b, tol=tol)

            if method in iterative_methods:
                n_iter_tot += n_iter

        t = time.time() - start_time
        print(f"{method}: Time: {t:.2e} s. Total iterations: {n_iter_tot}")


def main():
    run_and_time(n=int(1e4), N=100, tol=1e-6, methods=["cg", "gauss", "lu", "jacobi"])


if __name__ == "__main__":
    main()
