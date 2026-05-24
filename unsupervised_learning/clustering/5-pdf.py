#!/usr/bin/env python3
'''
PDF of a Gaussian distribution.
'''

import numpy as np


def pdf(X, m, S):
    '''
    Calculates the PDF of a Gaussian distribution.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
            - n is the number of data points
            - d is the number of dimensions for each data point
        m: numpy.ndarray of shape (d,) containing the mean of the distribution
        S: numpy.ndarray of shape (d, d) containing the covariance of the
        distribution
    Returns:
        P: numpy.ndarray of shape (n,) containing the PDF values for each data
        point, or None on failure
    '''

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0] or \
        S.shape[0] != S.shape[1]:
        return None

    n, d = X.shape

    # Check if covariance matrix is invertible by evaluating the determinant
    det_S = np.linalg.det(S)
    if det_S <= 0:
        return None

    S_inv = np.linalg.inv(S)

    # Calculate Gaussian PDF components
    norm_const = 1.0 / (np.power(2 * np.pi, d / 2) * np.sqrt(det_S))
    diff = X - m
    exponent = -0.5 * np.sum(diff @ S_inv * diff, axis=1)
    P = norm_const * np.exp(exponent)

    # Smooth out underflow values so extreme outliers default to
    # 1e-300 instead of 0.0
    P = np.maximum(P, 1e-300)

    return P
