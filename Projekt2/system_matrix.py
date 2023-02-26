import numpy as np
import scipy.sparse as sps


def system_matrix(n):
    A = (
        1 * np.diag(np.ones(n - 1), 1)
        + 1 * np.diag(np.ones(n - 1), -1)
        + 4 * np.diag(np.ones(n), 0)
    )
    A = sps.csr_matrix(A)
    return A
