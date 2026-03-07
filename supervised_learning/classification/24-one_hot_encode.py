#!/usr/bin/env python3
'''
one-hot encode
'''


import numpy as np


def one_hot_encode(Y, classes):
    '''
    converts a numeric label vector to one hot matrix
    Y - ndarray with shape (m, ) containing numeric label class
    classes - the maximum number of classes found in Y
    '''
    try:
        m = Y.shape[0]
        one_hot_matrix = np.zeros((classes, m), dtype=float)
        one_hot_matrix[Y, np.arange(m)] = 1.0
        return one_hot_matrix
    except Exception as e:
        return None
