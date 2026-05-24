#!/usr/bin/env python3
'''
K-means clustering.
'''

import numpy as np


def kmeans(X, k, iterations=1000):
    '''
    Performs K-means on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
            - n is the number of data points
            - d is the number of dimensions for each data point
        k: positive integer, the number of clusters
        iterations: positive integer, the maximum number of iterations
    Returns:
        C: numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster, or None on failure
        clss: numpy.ndarray of shape (n,) containing the index of the
        cluster in C that each data point belongs to, or None on failure
    '''
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    C = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), size=(k, d))
    clss = np.full(n, -1)

    # This is the single loop in the entire function
    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        new_clss = np.argmin(distances, axis=1)

        if np.array_equal(clss, new_clss):
            break
        clss = new_clss

        # Vectorized Centroid Update (No loop!)
        # We look up which points belong to which cluster index (0 to k-1)
        # If a cluster is empty, it keeps its previous centroid value
        C = np.array([X[clss == i].mean(axis=0) if np.any(clss == i) else C[i] for i in range(k)])

    return C, clss
