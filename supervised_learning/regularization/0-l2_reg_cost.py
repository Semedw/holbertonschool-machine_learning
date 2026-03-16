#!/usr/bin/env python3
'''
l2 (ridge) regularization
'''

import numpy as np
import tensorflow as tf


def l2_reg_cost(cost, lambd, weights, L, m):
    '''
    cost - the cost of the network without L2 regularization
    lambd - the regularization parameter
    weights - a dict of the weights and biases(ndarray) of the neural network
    L - the number of layers in the neural network
    m - the number of data points used
    '''
    sq_weights = 0
    for i in range(L):
        sq_weights += np.sum(np.square(weights[i]))

    l2_cost = (lambd / (2*m)) * sq_weights
    return l2_cost
