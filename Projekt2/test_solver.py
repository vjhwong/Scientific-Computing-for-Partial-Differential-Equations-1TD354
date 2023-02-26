import numpy as np
import numpy.linalg as nplg
import scipy.sparse.linalg as spsplg
from conjugate_gradient_solve import conjugate_gradient_solve
from project2_1 import jacobi_solve, jacobi_setup
from system_matrix import system_matrix as test_matrix


def test_solver(n=1000, N=3, method="lu", tol=1e-6):
    """Solves N nxn systems using {method} and compares with
    a direct solver (spsplg.solve())."""
    A = test_matrix(n)
    B = [np.random.rand(n) for _ in range(N)]

    if method == "lu":
        lu = spsplg.splu(A)
    elif method == "jacobi":
        Dinv, L_plus_U = jacobi_setup(A)

    for b in B:
        x_true = spsplg.spsolve(A, b)

        if method == "lu":
            x = lu.solve(b)
        elif method == "jacobi":
            x, n_iter = jacobi_solve(Dinv, L_plus_U, b, tol=tol)
        elif method == "cg":
            x, n_iter = conjugate_gradient_solve(A, b, tol=tol)

        if nplg.norm(x - x_true) / nplg.norm(x_true) > 10 * tol:
            print(f"Error! {method} yields an error larger than {10*tol:.2e}.")
