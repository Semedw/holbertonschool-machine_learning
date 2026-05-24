#!/usr/bin/env python3
"""Bayesian Information Criterion for GMM cluster selection."""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using BIC.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        kmin: positive integer, minimum number of clusters to check
        kmax: positive integer, maximum number of clusters to check
        iterations: positive integer, maximum number of EM iterations
        tol: non-negative float, EM tolerance threshold
        verbose: boolean, determines if EM output logs to stdout
    Returns:
        best_k: the best value for k based on BIC
        best_result: tuple containing (pi, m, S) of the best k GMM model
        l: numpy.ndarray of shape (kmax - kmin + 1,) with log likelihoods
        b: numpy.ndarray of shape (kmax - kmin + 1,) with BIC values
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    
    n, d = X.shape
    
    # Handle the default None initialization for kmax
    if kmax is None:
        kmax = n
        
    if not isinstance(kmin, int) or kmin <= 0 or kmin >= n:
        return None, None, None, None
    if not isinstance(kmax, int) or kmax <= 0 or kmax > n:
        return None, None, None, None
    if kmin > kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    k_best = []
    best_res = []
    logl_val = []
    bic_val = []
    
    for k in range(kmin, kmax + 1):
        pi, m, S, _, log_l = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        k_best.append(k)
        best_res.append((pi, m, S))
        logl_val.append(log_l)

        # Calculate number of independent parameters (p)
        cov_params = k * d * (d + 1) / 2.
        mean_params = k * d
        p = int(cov_params + mean_params + k - 1)

        # Formula for BIC calculation
        bic = p * np.log(n) - 2 * log_l
        bic_val.append(bic)

    bic_val = np.array(bic_val)
    logl_val = np.array(logl_val)
    best_val = np.argmin(bic_val)

    best_k = k_best[best_val]
    best_result = best_res[best_val]

    return best_k, best_result, logl_val, bic_val
