#!/usr/bin/env python3
'''
one-hot decode
'''


import numpy as np


def one_hot_decode(one_hot):
    '''
    decodes one hot
    one_hot - one-hot encoded ndarray with shape (classes, m)
    '''
    if len(one_hot.shape) != 2:
        return None
    Y = np.argmax(one_hot, axis=0)
    return Y
