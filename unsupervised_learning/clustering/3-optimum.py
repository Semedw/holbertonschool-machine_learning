#!/usr/bin/env python3
'''
K-optimum clustering.
'''

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    '''
    Calculates the optimum number of clusters for a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
            - n is the number of data points
            - d is the number of dimensions for each data point
        kmin: positive integer, the minimum number of clusters to check for
        kmax: positive integer, the maximum number of clusters to check for
        iterations: positive integer, the maximum number of iterations for
        K-means
    Returns:
        results: list containing the outputs of K-means for each cluster size
        d_vars: list containing the difference in variance from the smallest
        cluster size for each cluster size, or None on failure
    '''
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is not None and (not isinstance(kmax, int) or
                             kmax <= 0 or kmax <= kmin):
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    n, d = X.shape
    if kmax is None:
        kmax = n
    results = []
    d_vars = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        var = variance(X, C)
        if var is None:
            return None, None
        results.append((C, clss))
        d_vars.append(var)
    d_vars = [d_vars[0] - var for var in d_vars]
    return results, d_vars
