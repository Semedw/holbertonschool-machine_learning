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
    l2_costs = []
    for i in range(len(model.layers)):
        if hasattr(model.layers[i], 'kernel_regularizer'):
            l2_cost = model.layers[i].kernel_regularizer(model.layers[i].kernel)
            l2_costs.append(l2_cost)

    total_l2_cost = tf.add_n(l2_costs)

    return cost + total_l2_cost
