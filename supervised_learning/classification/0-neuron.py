#!/usr/bin/env python3
'''
writing neural network
'''


import numpy as np


class Neuron:
    '''
    neuron class
    '''

    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        self.W = np.random.randn(nx, shape=1)
        self.b = 0
        self.A = 0
