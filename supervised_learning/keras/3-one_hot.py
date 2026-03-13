#!/usr/bin/env python3
'''
one hot encoding the label vector
'''


import tensorflow.keras as K


def one_hot(labels, classes=None):
    '''
    one-hot encoding matrix
    '''
    mat = K.utils.to_categorical(labels-1, classes)
    return mat
