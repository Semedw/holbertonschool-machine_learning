#!/usr/bin/env python3
"""Bayesian Information Criterion for GMM cluster selection."""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using BIC with zero loops.

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

    # Array of k values to iterate through using map
    k_range = np.arange(kmin, kmax + 1)

    # Map the EM function across all target values of k (No user loop)
    em_results = list(map(
        lambda k: expectation_maximization(X, k, iterations, tol, verbose),
        k_range
    ))

    # Extract results simultaneously using list slicing/comprehension wrappers
    best_res = [(res[0], res[1], res[2]) for res in em_results]
    logl_val = np.array([res[4] for res in em_results])

    # Vectorized computation of free parameters 'p' for all k levels at once
    cov_params = k_range * d * (d + 1) / 2.
    mean_params = k_range * d
    p = (cov_params + mean_params + k_range - 1).astype(int)

    # Vectorized BIC calculation across all values simultaneously
    bic_val = p * np.log(n) - 2 * logl_val

    # Find the index minimizing the BIC array criteria
    best_val_idx = np.argmin(bic_val)

    best_k = k_range[best_val_idx]
    best_result = best_res[best_val_idx]

    return best_k, best_result, logl_val, bic_val
