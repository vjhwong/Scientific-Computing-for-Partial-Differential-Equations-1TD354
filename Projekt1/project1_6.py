import numpy as np
from scipy.sparse import kron, csc_matrix, eye, vstack, bmat, identity
from scipy.sparse.linalg import inv
from math import sqrt, ceil
import operators as ops
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import rungekutta4 as rk4


def f(x, y, sigma, x_0=0, y_0=0):
    return np.exp(-((x - x_0) ** 2) / (sigma**2) - ((y - y_0) ** 2) / (sigma**2))


def run_simulation(mx=200, my=100):
    """
    Solves the wave equation in 2D using finite differences
    and Runge-Kutta 4.

    Method parameters:
    mx:     Number of intervals in x, integer > 15.
    mx + 1: Number of grid points in x
    my: Number of intervals in y
    my + 1: Number of grid points in y
    """

    # Model parameters
    c = 1  # wave speed
    T = 3  # end time
    sigma = 0.05
    x_west = -1  # west boundary
    x_east = 1  # east boundary
    y_south = -0.5  # south boundary
    y_north = 0.5  # north boundary
    tau_l = c**2  # tau left
    tau_r = -(c**2)  # tau right

    # Space discretization
    hx = 2 / mx
    hy = 2 / my
    xvec = np.linspace(x_west, x_east, mx + 1)
    yvec = np.linspace(y_north, y_south, my + 1)
    X, Y = np.meshgrid(xvec, yvec)

    # Create SBP Operators
    _, HI, _, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_6th(mx + 1, hx)
    D_x = (c**2) * D2 + tau_l * HI @ e_l.T @ d1_l + tau_r * HI @ e_r.T @ d1_r

    _, HI, _, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_6th(my + 1, hy)
    D_y = (c**2) * D2 + tau_l * HI @ e_l.T @ d1_l + tau_r * HI @ e_r.T @ d1_r

    # Define phi and phi_t
    phi = f(X, Y, sigma).flatten()
    phi_t = np.zeros((mx + 1) * (my + 1))

    # Define solution vector
    w = np.concatenate((phi, phi_t))

    # Create identity matrices
    I_x = identity(mx + 1)
    I_y = identity(my + 1)

    # Time discretization
    ht_try = 0.1 * hx / c
    mt = int(np.ceil(T / ht_try) + 1)  # round up so that (mt-1)*ht = T
    tvec, ht = np.linspace(0, T, mt, retstep=True)

    # Initialize time variable and solution vector
    t = 0
    # w = [w1, w2]^T
    w = np.array([phi, phi_t]).reshape(-1, 1)  # Make column vector

    # Define A
    A = kron(I_y, D_x) + kron(D_y, I_x)

    # Define M to be used in rhs
    M = kron(np.array([[0, 1], [0, 0]]), identity((mx + 1) * (my + 1))) + kron(
        np.array([[0, 0], [1, 0]]), A
    )

    # Define right-hand-side function
    def rhs(w):
        return M @ w

    # Loop over all time steps
    for _ in range(mt - 1):

        # Take one step with the fourth order Runge-Kutta method.
        w, t = rk4.step(rhs, w, t, ht)

    return X, Y, w[: ((mx + 1) * (my + 1))]


def exact_solution(x, y, t, c):
    k_x = 2 * np.pi
    k_y = 2 * np.pi
    omega = c * np.sqrt((k_x**2) + (k_y**2))
    return np.cos(k_x * x) * np.cos(k_y * y) * np.cos(omega * t)


def color_plot_solution(X, Y, Z):
    fig, ax = plt.subplots()
    plt.xlabel("x")
    plt.ylabel("y")
    ax.pcolor(X, Y, Z.reshape(X.shape), shading="nearest")
    # Add a color bar which maps values to colors.
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def surface_plot_solution(X, Y, Z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        X, Y, Z.reshape(X.shape), cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter("{x:.02f}")
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def main():
    X, Y, Z = run_simulation(mx=200, my=100)
    color_plot_solution(X, Y, Z)
    surface_plot_solution(X, Y, Z)


if __name__ == "__main__":
    main()
