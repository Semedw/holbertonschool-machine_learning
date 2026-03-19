#!/usr/bin/env python3
'''
create a new layer with dropout
'''

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    '''
    prev - a tensor containing the output of the previous layer
    n - the number of nodes the new layer should contain
    activation - the activation function for the new layer
    keep_prob - the probability that a node will be kept
    training - a boolean indicating whether the model is in training mode

    Returns: the output of the new layer
    '''
    layer = tf.keras.layers.Dense(
        n,
        activation=activation,
        kernel_regularizer=tf.keras.layers.Dropout(keep_prob)
    )
    return layer[prev]
