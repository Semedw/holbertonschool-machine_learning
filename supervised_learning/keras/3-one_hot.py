#!/usr/bin/env python3
'''
one hot encoding the label vector
'''


import tensorflow.keras as K


def one_hot(labels, classes=None):
    '''
    one-hot encoding matrix
    '''
    mat = K.ops.one_hot(labels, classes, axis=-1)
    return mat
