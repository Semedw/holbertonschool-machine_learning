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
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    n, d = X.shape

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    C = np.random.uniform(X_min, X_max, (k, d))

    for i in range(iterations):
        centroids = np.copy(C)
        centroids_2 = C[:, np.newaxis]

        dist = np.sqrt(np.sum((X - centroids_2)**2, axis=2))
        clss = np.argmin(dist, axis=0)

        for c in range(k):
            if X[clss == c].size == 0:
                C[c] = np.random.uniform(X_min, X_max, size=(1, d))
            else:
                C[c] = X[clss == c].mean(axis=0)

        centroids_2 = C[:, np.newaxis]
        dist = np.sqrt(np.sum((X - centroids_2)**2, axis=2))
        clss = np.argmin(dist, axis=0)

        if (centroids == C).all():
            break
    return C, clss
