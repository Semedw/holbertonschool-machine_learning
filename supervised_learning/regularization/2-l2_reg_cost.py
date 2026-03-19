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
    reg_loss = [tf.add_n(lay.losses) for lay in model.layers if lay.losses]
    return cost + tf.stack(reg_loss)
