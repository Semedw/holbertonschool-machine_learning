#!/usr/bin/env python3
'''
normalizing matrixes
'''

import numpy as np


def normalize(X, m, s):
    '''
    X - the numpy.ndarray of shape (d, nx) to normalize
        d - the number of data points
        nx - the number of features
    m - a numpy.ndarray of shape (nx,) that contains 
        the mean of all features of X
    s - a numpy.ndarray of shape (nx,) that contains 
        the standard deviation of all features of X
    '''

    norm = np.linalg.norm(X)
    return norm
