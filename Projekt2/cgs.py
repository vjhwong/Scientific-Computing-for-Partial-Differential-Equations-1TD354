import numpy as np
import numpy.linalg as nplg


def conjugate_gradient_solve(A, b, x0=None, tol=1e-6):

    # If no initial guess supplied, use the zero vector
    if x0 is None:
        x_new = 0 * b

    # r: residual
    # p: search direction
    r = b - A @ x_new
    rho = nplg.norm(r) ** 2
    p = np.copy(r)
    err = 2 * tol
    n_iter = 0
    while err > tol:
        x = x_new
        w = A @ p
        Anorm_p_squared = np.dot(p, w)

        # If norm_A(p) is 0, we should have converged.
        if Anorm_p_squared == 0:
            break

        alpha = rho / Anorm_p_squared
        x_new = x + alpha * p
        r -= alpha * w
        rho_prev = rho
        rho = nplg.norm(r) ** 2
        p = r + (rho / rho_prev) * p
        err = nplg.norm(x_new - x) / nplg.norm(x_new)
        n_iter += 1

    return x_new, n_iter
