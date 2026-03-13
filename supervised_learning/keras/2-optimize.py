#!/usr/bin/env python3
'''
optimizing model
'''


import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    '''
    optimizing model
    network - is the model to optimize
    alpha - is the learning rate
    beta1 - the first Adam optimization parameter
    beta2 - the second Adam optimization parameter
    '''

    network = network.optimizers.Adam(
                learning_rate=alpha,
                beta_1=beta1,
                beta_2=beta2
                )
