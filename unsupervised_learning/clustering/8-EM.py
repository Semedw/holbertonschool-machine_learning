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
        return None, None, None, None, None

    # Run initial E-step to get starting log-likelihood
    g, log_likelihood = expectation(X, pi, m, S)
    if g is None or log_likelihood is None:
        return None, None, None, None, None

    if verbose:
        print("Log Likelihood after 0 iterations: {:.5f}".format(log_likelihood))

    # Save initial likelihood to check for early convergence differences
    l_old = log_likelihood

    # 2. Main EM Loop
    for i in range(iterations):
        # M-Step
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        # E-Step
        g, log_likelihood = expectation(X, pi, m, S)
        if g is None or log_likelihood is None:
            return None, None, None, None, None

        # Print layout matches: every 10 iterations
        if verbose and (i + 1) % 10 == 0:
            print("Log Likelihood after {} iterations: {:.5f}".format(i + 1, log_likelihood))

        # Check for convergence based on tolerance
        if abs(log_likelihood - l_old) <= tol:
            # If we break early and haven't just printed this iteration, print it now
            if verbose and (i + 1) % 10 != 0:
                print("Log Likelihood after {} iterations: {:.5f}".format(i + 1, log_likelihood))
            break
            
        l_old = log_likelihood

    # Return exactly 5 values to prevent the unpack ValueError
    return pi, m, S, g, log_likelihood
