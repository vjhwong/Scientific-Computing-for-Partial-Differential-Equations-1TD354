from laplace_equation import laplace_equation
import time

######################################################################################
##                                                                                  ##
##  Lab "Iterative methods", for the course "Scientific computing for PDEs"         ##
##  at Uppsala University.                                                          ##
##  Based on Matlab code given in the course "Scientific Computing 3"               ##
##                                                                                  ##
##  Part 2: Solves the Laplace equation on the unit square.                         ##
##                                                                                  ##
##  N      : number of interior grid points in each dimension                       ##
##  method : 'jacobi', 'gauss-seidel', or 'conjugate-gradient'                      ##
##                                                                                  ##
##  The code has been tested on the following versions:                             ##
##   - Python 3.8.16                                                                ##
##   - Numpy 1.24.1                                                                 ##
##   - Scipy 1.10.0                                                                 ##
##   - Matplotlib 3.6.3                                                             ##
##                                                                                  ##
######################################################################################


def time_function(function, N, method):
    tstart = time.perf_counter()
    function(N, method, show_plots=False)
    tstop = time.perf_counter()
    # print(f"Time for {method}: {tstop-tstart}")
    return tstop - tstart


def main():
    N = 100
    run_time_jacobi = time_function(laplace_equation, N, "jacobi")
    run_time_gs = time_function(laplace_equation, N, "gauss-seidel")
    run_time_cg = time_function(laplace_equation, N, "conjugate-gradient")

    print(f"Run time for Jacobi: {run_time_jacobi}")
    print(f"Run time for Gauss-Seidel: {run_time_gs}")
    print(f"Run time for Conjugate Gradient: {run_time_cg}")


if __name__ == "__main__":
    main()
