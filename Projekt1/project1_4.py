import numpy as np
import scipy.sparse.linalg as spsplg
import scipy.linalg as splg
import operators as ops
import matplotlib.pyplot as plt
import time
import rungekutta4 as rk4

# Initial data
n = 2
k = 2 * np.pi


def f(x):
    return np.cos(k * x)


def run_simulation(mx=100, order=2, show_animation=True):
    """Solves the advection equation using finite differences
    and Runge-Kutta 4.

    Method parameters:
    mx:     Number of grid points, integer > 15.
    order:  Order of accuracy, 2, 4, 6, 8, 10 or 12
    """

    # Model parameters
    c = 3  # wave speed
    T = np.pi  # end time
    xl = -1  # left boundary
    xr = 1  # right boundary
    L = xr - xl  # domain length

    # Space discretization
    hx = (xr - xl) / mx
    xvec = np.linspace(xl, xr - hx, mx)  # periodic, u(xl) = u(xr)
    n_points_x = len(xvec)

    if order == 2:
        _, HI, _, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_2nd(mx, hx)

    elif order == 4:
        _, HI, _, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_4th(mx, hx)

    elif order == 6:
        _, HI, _, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_6th(mx, hx)
    else:
        raise NotImplementedError("Order not implemented")

    tau_l = c**2  # tau left
    tau_r = -(c**2)  # tau right

    phi = f(xvec)
    phi_t = np.zeros(n_points_x)

    # Initialize solution vector
    # w = [w1, w2]^T
    w = np.array([phi, phi_t]).reshape(-1, 1)

    # Define right-hand-side function
    def rhs(u):
        # dw1/dt = w2
        # dw2/dt = rhs equation for v_tt
        u_t = np.zeros((2 * n_points_x, 1))
        Du = (
            (c**2) * D2 @ u[n_points_x:]
            + tau_l * HI @ e_l.T @ d1_l @ u[n_points_x:]
            + tau_r * HI @ e_r.T @ d1_r @ u[n_points_x:]
        )
        u_t[:n_points_x] = Du
        u_t[n_points_x:] = u[:n_points_x]
        return u_t

    # Time discretization
    ht_try = 0.1 * hx / c
    mt = int(np.ceil(T / ht_try) + 1)  # round up so that (mt-1)*ht = T
    tvec, ht = np.linspace(0, T, mt, retstep=True)

    # Initialize time variable and solution vector
    t = 0

    # Loop over all time steps
    for _ in range(mt - 1):

        # Take one step with the fourth order Runge-Kutta method.
        w, t = rk4.step(rhs, w, t, ht)

    return w, T, xvec, hx, L, c


def exact_solution(t, xvec, L, c):
    T1 = L / c  # Time for one lap
    t_eff = (t / T1 - np.floor(t / T1)) * T1  # "Effective" time, using periodicity
    u_exact = f(xvec - c * t_eff)
    return u_exact.reshape(-1, 1)


def l2_norm(vec, h):
    return np.sqrt(h) * np.sqrt(np.sum(vec**2))


def compute_error(u, u_exact, hx):
    """Compute discrete l2 error"""
    error_vec = u - u_exact
    relative_l2_error = l2_norm(error_vec, hx) / l2_norm(u_exact, hx)
    return relative_l2_error


def error(m, order):
    u, T, xvec, hx, L, c = run_simulation(m, order, show_animation=False)
    u = u[:m]
    u_exact = exact_solution(T, xvec, L, c)
    error = compute_error(u, u_exact, hx)
    return (hx, error)


def convergence_study(m_vec, order_vec):
    error_vec = np.zeros((len(order_vec), len(m_vec)))
    h_vec = np.zeros((len(order_vec), len(m_vec)))

    for i, order in enumerate(order_vec):
        print(f"Order: {order}")
        for j, m in enumerate(m_vec):
            print(f"m: {m}")
            h_vec[i, j], error_vec[i, j] = error(m, order)
    return (h_vec, error_vec)


def main():
    # m_vec = np.array([25, 50, 100, 200, 400])
    # order_vec = np.array([2, 4, 6])
    m_vec = np.array([25, 50, 100])
    order_vec = np.array([2])
    h_vec, error_vec = convergence_study(m_vec, order_vec)
    print(error_vec)


if __name__ == "__main__":
    main()
