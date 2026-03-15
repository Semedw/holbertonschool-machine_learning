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
        scale=2.0, 
        mode='fan_avg'
    )

    # Base Dense layer
    # Note: use_bias=True is default, but BN will effectively 
    # override the Dense bias via its own 'beta' parameter.
    dense_layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer,
        use_bias=True
    )
    
    Z = dense_layer(prev)

    # Batch Normalization layer
    # gamma_initializer='ones' and beta_initializer='zeros' are defaults
    batch_norm_layer = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        gamma_initializer='ones',
        beta_initializer='zeros'
    )
    
    Z_norm = batch_norm_layer(Z)

    # Apply activation function if provided
    if activation is not None:
        return activation(Z_norm)

    return Z_norm