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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''
        Weight getter
        '''
        return self.__W

    @property
    def b(self):
        '''
        Bias getter
        '''
        return self.__b

    @property
    def A(self):
        '''
        Activated neuron getter
        '''
        return self.__A

    def forward_prop(self, X):
        '''
        calculating forward propagation
        '''
        Z = np.matmul(X, self.__W) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
