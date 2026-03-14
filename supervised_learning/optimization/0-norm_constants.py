#!/usr/bin/env python3
'''
normalizing constants
'''

import numpy as np
# import tensorflow as tf


def normalization_constants(X):
    '''
    X - ndarray of shape (m, nx):
        nx - the number of features
        m - the number of data points
    '''
    mean = np.mean(X, axis=0)
    stddev = np.std(X, axis=0)
    return mean, stddev
