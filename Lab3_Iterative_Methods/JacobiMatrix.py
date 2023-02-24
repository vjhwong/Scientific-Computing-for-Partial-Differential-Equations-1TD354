import numpy as np
import scipy.sparse as spsp


def JacobiMatrix(A):
    """
    Construct the Jacobi Matrix for A
    """
    M = spsp.diags(A.diagonal())
    Minv = spsp.diags(1 / A.diagonal())
    G = Minv * (M - A)
    return G
