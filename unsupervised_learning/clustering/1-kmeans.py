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
    
    # Initialize centroids (Your initialization is correct!)
    C = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), size=(k, d))
    
    for _ in range(iterations):
        # 1. Calculate distances and assign clusters
        # A more standard subtraction matrix helps avoid axis=2 floating variance
        distances = np.sqrt(((X - C[:, np.newaxis]) ** 2).sum(axis=2))
        new_clss = np.argmin(distances, axis=0)
        
        # 2. Update centroids using the new classifications
        C_old = C.copy()
        for i in range(k):
            if np.any(new_clss == i):
                C[i] = X[new_clss == i].mean(axis=0)
                
        # 3. Check for convergence based on centroids not moving
        if np.array_equal(C_old, C):
            break
            
    # One final assignment to make sure clss matches the final centroids perfectly
    distances = np.sqrt(((X - C[:, np.newaxis]) ** 2).sum(axis=2))
    clss = np.argmin(distances, axis=0)
    
    return C, clss
