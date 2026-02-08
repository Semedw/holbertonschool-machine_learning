#!/usr/bin/env python3
"""
finding likelihood
"""

import numpy as np


def likelihood(x, n, P):
    """
    inside the function
    """
    res = np.array([])
    if not isinstance(n, int) or  n<=0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError('x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    for i in P:
        if i < 0 or i > 1:
            raise ValueError('All values in P must be in the range [0, 1]')
    for i in P:
        lk = i * n / x
        res.append(lk)
    return res
