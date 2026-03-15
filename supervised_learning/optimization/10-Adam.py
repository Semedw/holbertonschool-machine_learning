#!/usr/bin/env python3
'''
adam optimization
'''

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    '''
    Create the training operation for a neural network in tensorflow using
    the Adam optimization algorithm

    Parameters:
    loss: loss of the network
    alpha: learning rate
    beta1: weight for the first moment
    beta2: weight for the second moment
    epsilon: small number to avoid division by zero

    Returns:
    Adam optimization operation
    '''
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
