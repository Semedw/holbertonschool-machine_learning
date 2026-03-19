#!/usr/bin/env python3
'''
forward propagation with dropout
'''

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prop):
    '''
    X - a ndarray of shape (nx, m) containing the input data for the network
        - nx is the number of input features
        - m is the number of data points 
    weights - a dictionary of the weights and biases of the neural network
    L - the number of layers in the network
    keep_prop - the probability that a node will be kept
    '''
    cache = {}
    cache['A0'] = X
    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        Z = np.matmul(W, A_prev) + b
        A = 1 / (1 + np.exp(-Z))
        D = np.random.rand(A.shape[0], A.shape[1]) < keep_prop
        A *= D
        A /= keep_prop
        cache['A' + str(i)] = A
        cache['D' + str(i)] = D
    return cache
