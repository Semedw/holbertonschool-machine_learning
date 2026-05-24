#!/usr/bin/env python3
'''
Expectation step in the EM algorithm for a GMM.
'''

import numpy as np


def expectation(X, pi, m, S):
    '''
    Performs the expectation step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
            - n is the number of data points
            - d is the number of dimensions for each data point
        pi: numpy.ndarray of shape (k,) containing the priors for each cluster
        m: numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster
        S: numpy.ndarray of shape (k, d, d) containing the covariance
        matrices for each cluster
    Returns:
        g: numpy.ndarray of shape (k, n) containing the posterior probabilities
        for each data point in each cluster, or None on failure
        l: total log likelihood, or None on failure
    '''
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if (X.shape[1] != m.shape[1] or 
            m.shape[0] != pi.shape[0] or 
            S.shape[0] != pi.shape[0] or 
            S.shape[1] != m.shape[1] or 
            S.shape[2] != m.shape[1]):
        return None, None
    n, d = X.shape
    k = pi.shape[0]
    g = np.zeros((k, n))
    for i in range(k):
        diff = X - m[i]
        inv_S = np.linalg.inv(S[i])
        exponent = -0.5 * np.sum(diff @ inv_S * diff, axis=1)
        coeff = pi[i] / np.sqrt((2 * np.pi) ** d * np.linalg.det(S[i]))
        g[i] = coeff * np.exp(exponent)
    g_sum = np.sum(g, axis=0)
    g_sum[g_sum == 0] = 1e-300
    g /= g_sum
    l = np.sum(np.log(g_sum))
    return g, l
