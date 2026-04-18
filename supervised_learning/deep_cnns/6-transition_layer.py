#!/usr/bin/env python3
'''
transition layer
'''

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    '''
    X - the output from the previous layer
    nb_filters - an integer representing the number of filters in X
    compression - the compression factor for the transition layer
    Your code should implement compression as used in DenseNet-C
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    All convolutions should be preceded by Batch Normalization and
    a rectified linear activation (ReLU), respectively

    Returns: The output of the transition layer and
    the number of filters within the output, respectively
    '''
    init = K.initializer.he_normal(seed=0)
    filters = int(compression * nb_filters)

    batch_normal = K.layers.BatchNormalization()(X)
    relu = K.layers.ReLU()(batch_normal)
    conv2d = K.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            kernel_initializer=init,
            padding='same'
            )(relu)
    avgpool = K.layers.AveragePooling2D()(conv2d)

    return avgpool, filters
