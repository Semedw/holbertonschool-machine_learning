#!/usr/bin/env python3
"""Initialize the project."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initialize cluster centroids for K-means.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
            - n is the number of data points
            - d is the number of dimensions for each data point
        k: positive integer, the number of clusters
    Returns:
        pi, m, S, or None, None, None on failure
            - pi is a numpy.ndarray of shape (k,) containing the priors for each cluster, initialized evenly
            - m is a numpy.ndarray of shape (k, d) containing the centroid means for each cluster, initialized with K-means
            - S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices for each cluster, initialized as identity matrices
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None

    n, d = X.shape
    phi = np.ones(k) / k
    m, _ = kmeans(X, k)
    S = np.tile(np.identity(d), (k, 1)).reshape(k, d, d)
    return phi, m, S
