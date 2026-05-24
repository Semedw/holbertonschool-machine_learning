#!/usr/bin/env python3
"""
Calculates the total intra-cluster variance for a data set.
"""

import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
            - n is the number of data points
            - d is the number of dimensions for each data point
        C: numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster
    Returns:
        var: total intra-cluster variance, or None on failure
    """
    # no loops allowed
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    n, d = X.shape
    centroids_2 = C[:, np.newaxis]
    dist = np.sqrt(np.sum((X - centroids_2) ** 2, axis=2))
    min_dist = np.min(dist, axis=0)

    variance = np.sum(min_dist ** 2)
    return variance
