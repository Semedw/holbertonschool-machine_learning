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
    
    # 1. Initialize Centroids uniformly within data bounds
    C = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), size=(k, d))
    
    # Initialize labels with a placeholder that won't trigger premature convergence
    clss = np.full(n, -1)

    # The ONLY loop allowed in the entire function
    for _ in range(iterations):
        # 2. Distance Matrix Calculation
        # Transposing the subtraction resolves axis broadcasting precision quirks
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        new_clss = np.argmin(distances, axis=1)

        # 3. Check for Convergence
        if np.array_equal(clss, new_clss):
            break
        clss = new_clss

        # 4. Fully Vectorized Centroid Update (Zero Loops)
        # Count occurrences of each cluster label
        counts = np.bincount(clss, minlength=k)[:, np.newaxis]
        
        # Sum coordinates belonging to each cluster using advanced indexing
        sums = np.zeros((k, d))
        np.add.at(sums, clss, X)
        
        # Update centroids, avoiding division-by-zero for empty clusters
        C = np.where(counts > 0, sums / counts, C)

    return C, clss
