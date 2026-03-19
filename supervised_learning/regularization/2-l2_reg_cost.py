#!/usr/bin/env python3
'''
l2 regularization cost function
'''

import tensorflow as tf


def l2_reg_cost(cost, model):
    '''
    cost - tensor containing the cost of the network without L2 regularization
    model - a Keras model that includes layers with L2 regularization
    
    Returns: a tensor containing the total cost for each
            layer of the network, accounting for L2 regularization
    '''
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            200,
            activation = 'tanh',
            kernel_regularizer = tf.keras.regularizers.L2(0.01)),

        tf.keras.layers.Dense(
            10,
            activation = 'sigmoid',
            kernel_regularizer = tf.keras.regularizers.L2(0.01))
    ])

    return model