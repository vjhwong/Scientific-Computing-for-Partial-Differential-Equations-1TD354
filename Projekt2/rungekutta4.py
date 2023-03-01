"""4th order Runge-Kutta time-stepping.
This module solves ODEs of the form dv/dt = f(v).
"""


def FEM_step(f, A, v, M, method, t, dt):
    """Take one RK4 step. Return updated solution and time.
    f: Right-hand-side function: dv/dt = f(v)
    v: current solution
    t: current time
    dt: time step
    """

    # Compute rates k1-k4
    k1 = dt * f(A, v, M, method)
    k2 = dt * f(A, v + 0.5 * k1, M, method)
    k3 = dt * f(A, v + 0.5 * k2, M, method)
    k4 = dt * f(A, v + k3, M, method)

    # Update solution and time
    v = v + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    t = t + dt

    return v, t
