#!/usr/bin/env python3
'''
mini batch gradient descent
'''


import numpy as np


def create_mini_batches(X, Y, batch_size):
    '''
    X is the first numpy.ndarray of shape (m, nx) to shuffle
        m is the number of data points
        nx is the number of features in X

    Y is the second numpy.ndarray of shape (m, ny) to shuffle
        m is the same number of data points as in X
        ny is the number of features in Y

    batch_size is the number of data points in a batch

    Returns: a list of tuples, each tuple is (X_batch, Y_batch)
    '''

    m = X.shape[0]
    mini_batches = []
    for i in range(0, m, batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches