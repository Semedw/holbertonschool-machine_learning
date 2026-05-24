#!/usr/bin/env python3
'''
maximization step in the EM algorithm for a GMM.
'''

import numpy as np


def maximization(X, g):
    '''
    Performs the maximization step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
            - n is the number of data points
            - d is the number of dimensions for each data point
        g: numpy.ndarray of shape (k, n) containing the posterior probabilities
        for each data point in each cluster
    Returns:
        pi: numpy.ndarray of shape (k,) containing the updated priors for each
        cluster, or None on failure
        m: numpy.ndarray of shape (k, d) containing the updated centroid means
        for each cluster, or None on failure
        S: numpy.ndarray of shape (k, d, d) containing the updated covariance
        matrices for each cluster, or None on failure
    '''
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None
    if not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None
    n, d = X.shape
    k = g.shape[0]
    pi = np.sum(g, axis=1) / n
    m = np.dot(g, X) / np.sum(g, axis=1)[:, np.newaxis]
    S = np.zeros((k, d, d))
    for i in range(k):
        X_centered = X - m[i]
        S[i] = np.dot(g[i] * X_centered.T, X_centered) / np.sum(g[i])
    return pi, m, S
