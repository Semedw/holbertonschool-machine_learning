#!/usr/bin/env python3
'''
batch normalization upgraded
'''

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    '''
    creates a batch normalization layer for a neural network in tensorflow

    Parameters:
    prev: output from the previous layer
    n: number of nodes in the layer to be created
    activation: activation function that should be used on the output of the
    layer

    Returns:
    The activated output of the batch normalization layer
    '''
    initializer = tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in')

    # Create a dense layer with no activation function and the specified
    # initializer
    dense_layer = tf.keras.layers.Dense(
            n,
            activation=None,
            kernel_initializer=initializer)

    # Apply the dense layer to the input
    Z = dense_layer(prev)

    # Create a batch normalization layer and apply it to Z
    batch_norm_layer = tf.keras.layers.BatchNormalization()
    Z_norm = batch_norm_layer(Z)

    # Apply the specified activation function to the normalized output
    if activation is not None:
        activated_output = tf.keras.activations.get(activation)(Z_norm)
        return activated_output

    return Z_norm
