#!/usr/bin/env python3
'''
Gradient Descent with L2 Regularization
'''

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambd, L):
    '''
    Y - a one-hot numpy.ndarray of shape (classes, m) that contains the correct labels for the data
        - classes: the number of classes
        - m: the number of data points
    
    weights - a dictionary of the weights and biases of the neural network
    cache - a dictionary of the outputs of each layer of the neural network
    alpha - the learning rate
    lambd - the L2 regularization parameter
    L - the number of layers of the network
    '''

    for i in range(1, L+1):
        weights[f'W{i}'] = weights[f'W{i}'] - alpha * cache[i-1]
    