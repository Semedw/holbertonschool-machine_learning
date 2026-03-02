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
        self.nx = nx

    W = 0.2 * np.random.rand(1, 784) - 0.1
    b = 0
    A = 0
