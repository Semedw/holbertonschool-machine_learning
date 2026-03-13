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

    model = K.models.Sequential()  # this means: layer 1 > layer 2 > output

    for i in range(len(layers)):
        if i == 0:  # specifying input shape only in the first layer
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha),
                input_shape=(nx,)
        ))
        else:
            model.add(K.layers.Dense(
                    layers[i],
                    activation=activations[i],
                    kernel_regularizer=K.regularizers.l2(lambtha)
            ))

        # Add dropout except the last layer
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
