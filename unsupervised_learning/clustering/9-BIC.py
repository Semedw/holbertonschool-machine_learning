#!/usr/bin/env python3
'''
BIC calculation for a GMM.
'''

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    '''
    Calculates the BIC for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
            - n is the number of data points
            - d is the number of dimensions for each data point
        k: positive integer, the number of clusters
        iterations: positive integer, the maximum number of iterations for the
        algorithm
        tol: non-negative float, the tolerance of the log likelihood, used to
        determine early stopping of the algorithm based on the change in log
        likelihood of the model
        verbose: boolean, determines if the log likelihood should be printed
        during each iteration of the algorithm
    Returns:
        BIC: the BIC of the model, or None on failure
    '''
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0 or X.shape[0]:
        return None
    if not isinstance(iterations, int) or iterations <= 0:
        return None
    if not isinstance(tol, float) or tol < 0:
        return None
    if not isinstance(verbose, bool):
        return None

    k_best = []
    best_res = []
    logl_val = []
    bic_val = []
    n, d = X.shape
    for k in range(kmin, kmax + 1):
        pi, m, S, _, log_l = expectation_maximization(X, k, iterations, tol,
                                                      verbose)
        k_best.append(k)
        best_res.append((pi, m, S))
        logl_val.append(log_l)

        cov_params = k * d * (d + 1) / 2.
        mean_params = k * d
        p = int(cov_params + mean_params + k - 1)

        bic = p * np.log(n) - 2 * log_l
        bic_val.append(bic)

    bic_val = np.array(bic_val)
    logl_val = np.array(logl_val)
    best_val = np.argmin(bic_val)

    k_best = k_best[best_val]
    best_res = best_res[best_val]

    return k_best, best_res, logl_val, bic_val
