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
        samples = Y.size
        one_hot_matrix = np.zeros((samples, classes), dtype=float)
        one_hot_matrix[np.arange(samples), Y] = 1.0
        return one_hot_matrix
    except Exception as e:
        return None
    
