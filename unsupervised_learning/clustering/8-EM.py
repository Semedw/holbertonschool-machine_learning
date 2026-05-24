#!/usr/bin/env python3
'''
EM algorithm for a GMM.
'''

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    '''
    Performs the EM algorithm for a GMM.

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
        pi: numpy.ndarray of shape (k,) containing the priors for each cluster,
        or None on failure
        m: numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster, or None on failure
        S: numpy.ndarray of shape (k, d, d) containing the covariance
        matrices for each cluster, or None on failure
        g: numpy.ndarray of shape (k, n) containing the posterior probabilities
        for each data point in each cluster, or None on failure
    '''
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None
    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None
    g, log = expectation(X, pi, m, S)
    if g is None or log is None:
        return None, None, None, None
    if verbose:
        print('Log Likelihood after initialization: {}'.format(log))
    for i in range(iterations):
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None
        g, log_new = expectation(X, pi, m, S)
        if g is None or log_new is None:
            return None, None, None, None
        if verbose:
            print('Log Likelihood after {} iterations: {}'.format(i + 1, log_new))
        if abs(log_new - log) <= tol:
            break
        log = log_new
    return pi, m, S, g
