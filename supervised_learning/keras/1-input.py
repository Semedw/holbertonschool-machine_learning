#!/usr/bin/env python3
'''
building neural network with keras library
'''


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''
    building model
    nx - the number of input features
    layers - a list containing the number of nodes in each layer
    activations - list containing the activation functions used for each layer
    lambtha - the L2 regularization parameter
    keep_prob - the probability that a node will be kept for dropout
    '''

    inputs = K.Input(shape=(nx, ))
    x = inputs

    for i in range(len(layers)):

        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)

        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.models.Model(inputs=inputs, outputs=x)
    return model
