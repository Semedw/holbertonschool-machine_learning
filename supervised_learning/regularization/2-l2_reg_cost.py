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
    reg_cost = cost
    for layer in model.layers:
        for loss in layer.losses:
            reg_cost += loss
    return reg_cost