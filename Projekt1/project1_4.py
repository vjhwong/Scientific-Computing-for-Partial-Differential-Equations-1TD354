import numpy as np
import scipy.sparse.linalg as spsplg
import scipy.linalg as splg
import operators as ops
import matplotlib.pyplot as plt
import time
import rungekutta4 as rk4

# Initial data
n = 2
k = n * np.pi


def f(x):
    return np.cos(k * x)


def run_simulation(mx=100, order=2):
    """
    Solves the advection equation using finite differences
    and Runge-Kutta 4.

    Method parameters:
    mx:     Number of intervals, integer > 15.
    order:  Order of accuracy, 2, 4, 6, 8, 10 or 12
    """

    # Model parameters
    c = 3  # wave speed
    T = np.pi  # end time
    xl = -1  # left boundary
    xr = 1  # right boundary
    L = xr - xl  # domain length
    tau_l = c**2  # tau left
    tau_r = -(c**2)  # tau right

    # Space discretization
    hx = (xr - xl) / mx
    xvec = np.linspace(xl, xr, mx + 1)
    n_points_x = len(xvec)

    if order == 2:
        _, HI, _, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_2nd(n_points_x, hx)
    elif order == 4:
        _, HI, _, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_4th(n_points_x, hx)
    elif order == 6:
        _, HI, _, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_6th(n_points_x, hx)
    else:
        raise NotImplementedError("Order not implemented")

    # Define phi and phi_t
    phi = f(xvec)
    phi_t = np.zeros(n_points_x)

    # Define right-hand-side function
    def rhs(w):
        w_t = np.zeros((len(w), 1))
        Dw = (
            (c**2) * D2 @ w[:n_points_x]
            + tau_l * HI @ e_l.T @ d1_l @ w[:n_points_x]
            + tau_r * HI @ e_r.T @ d1_r @ w[:n_points_x]
        )
        # dw1/dt = w2
        w_t[n_points_x:] = Dw
        # dw2/dt = rhs equation for v_tt
        w_t[:n_points_x] = w[n_points_x:]
        return w_t

    # Time discretization
    ht_try = 0.1 * hx / c
    mt = int(np.ceil(T / ht_try) + 1)  # round up so that (mt-1)*ht = T
    tvec, ht = np.linspace(0, T, mt, retstep=True)

    # Initialize time variable and solution vector
    t = 0
    # w = [w1, w2]^T
    w = np.array([phi, phi_t]).reshape(-1, 1)  # Make column vector

    # Loop over all time steps
    for tidx in range(mt - 1):

        # Take one step with the fourth order Runge-Kutta method.
        w, t = rk4.step(rhs, w, t, ht)

    # Return first half of w vector which
    return w, T, xvec, hx, L, c


def exact_solution(x, t, c, n):
    k = n * np.pi
    omega = c * k
    return np.cos(k * x) * np.cos(omega * t)


def l2_norm(vec, h):
    return np.sqrt(h) * np.sqrt(np.sum(vec**2))


def compute_error(u, u_exact, hx):
    """Compute discrete l2 error"""
    error_vec = u.flatten() - u_exact
    relative_l2_error = l2_norm(error_vec, hx) / l2_norm(u_exact, hx)
    return relative_l2_error


def convergence_study(order_vec, m_vec):
    h_vec = np.zeros((len(order_vec), len(m_vec)))
    error_vec = np.zeros((len(order_vec), len(m_vec)))

    for i, order in enumerate(order_vec):
        for j, m in enumerate(m_vec):
            print(f"Order: {order}, m: {m}")
            u, T, xvec, hx, L, c = run_simulation(m, order)
            u = u[: m + 1]
            u_exact = exact_solution(xvec, T, c, n)
            error = compute_error(u, u_exact, hx)
            h_vec[i, j] = hx
            error_vec[i, j] = error
    return h_vec, error_vec


def plot_convergence(h_vec, error_vec):
    second_order_ref = [
        (error_vec[0, 0] / (2 ** (2 * i))) for i in range(error_vec.shape[1])
    ]
    fourth_order_ref = [
        (error_vec[1, 0] / (2 ** (4 * i))) for i in range(error_vec.shape[1])
    ]
    sixth_order_ref = [
        (error_vec[2, 0] / (2 ** (6 * i))) for i in range(error_vec.shape[1])
    ]
    fig, ax = plt.subplots()
    ax.loglog(
        h_vec[0, :],
        second_order_ref,
        color="mediumturquoise",
        linestyle="--",
        label="2nd order reference",
    )
    ax.loglog(h_vec[0, :], error_vec[0, :], color="mediumturquoise", label="2nd order")
    ax.loglog(
        h_vec[1, :],
        fourth_order_ref,
        color="khaki",
        linestyle="--",
        label="4th order reference",
    )
    ax.loglog(h_vec[1, :], error_vec[1, :], color="khaki", label="4th order")
    ax.loglog(
        h_vec[2, :],
        sixth_order_ref,
        color="coral",
        linestyle="--",
        label="6th order reference",
    )
    ax.loglog(h_vec[2, :], error_vec[2, :], color="coral", label="6th order")
    ax.set_xlabel("Step size [h]")
    ax.set_ylabel("Error")
    plt.xticks(h_vec[0, :])
    ax.grid(visible=True, axis="both")
    plt.legend()
    plt.show()


def main():
    m_vec = [25, 50, 100, 200, 400]
    order_vec = [2, 4, 6]

    h_vec, error_vec = convergence_study(order_vec, m_vec)
    plot_convergence(h_vec, error_vec)


if __name__ == "__main__":
    main()
