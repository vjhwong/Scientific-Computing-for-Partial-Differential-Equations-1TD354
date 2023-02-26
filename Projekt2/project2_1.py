from test_solve import test_solver


def main():
    test_solver(n=1000, N=3, tol=1e-6, method="lu")
    test_solver(n=1000, N=3, tol=1e-6, method="cg")
    test_solver(n=1000, N=3, tol=1e-6, method="jacobi")


if __name__ == "__main__":
    main()
