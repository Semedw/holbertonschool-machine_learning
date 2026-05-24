#!/usr/bin/env python3
"""Initialize the project."""

import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for K-means.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
            - n is the number of data points
            - d is the number of dimensions for each data point
        k: positive integer, the number of clusters
    Returns:
        C: numpy.ndarray of shape (k, d) containing the initialized
        centroids for each cluster, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    centroids = np.random.uniform(low, high, size=(k, X.shape[1]))

    return centroids
