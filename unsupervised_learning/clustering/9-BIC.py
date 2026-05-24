#!/usr/bin/env python3
'''
BIC calculation for a GMM.
'''

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, k, iterations=1000, tol=1e-5, verbose=False):
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
    n, d = X.shape
    pi, m, S, g = expectation_maximization(X, k, iterations, tol, verbose)
    if pi is None or m is None or S is None or g is None:
        return None
    p = k * d * (d + 1) / 2 + k * d + k - 1
    loglikelihood = np.sum(np.log(np.sum(pi[:, np.newaxis] * g, axis=0)))
    BIC = p * np.log(n) - 2 * loglikelihood
    return BIC
