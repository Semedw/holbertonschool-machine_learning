#!/usr/bin/env python3
'''
deep neural network
'''


import numpy as np


class DeepNeuralNetwork:
    '''
    building a deep neural network
    '''

    def __init__(self, nx, layers):
        '''
        nx - the number of input features -> int
        layers - the number of nodes in eahc layer of the network -> List
        e.g. layers[0] represents the the number of nodes in the first layer
        '''

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)  # the number of layers in the neural network
        self.__cache = {}  # to hold all intermediary values of the network
        self.__weights = {}
        for lay in range(self.L):
            if layers[lay] < 1 or type(layers[lay]) is not int:
                raise TypeError("layers must be a list of positive integers")
            self.weights["b" + str(lay + 1)] = np.zeros((layers[lay], 1))
            if lay == 0:
                He_val = np.random.randn(layers[lay], nx) * np.sqrt(2 / nx)
                self.weights["W" + str(lay + 1)] = He_val
            if lay > 0:
                He_val1 = np.random.randn(layers[lay], layers[lay - 1])
                He_val2 = np.sqrt(2 / layers[lay - 1])
                He_val3 = He_val1 * He_val2
                self.weights["W" + str(lay + 1)] = He_val3

    @property
    def L(self):
        '''
        L getter
        '''
        return self.__L

    @property
    def cache(self):
        '''
        cache getter
        '''
        return self.__cache

    @property
    def weights(self):
        '''
        weights getter
        '''
        return self.__weights
    
    @staticmethod
    def sigmoid(x):
        '''
        sigmoid function
        '''
        return 1 / (1 + np.exp(-x))

    def forward_prop(self, X):
        '''
        forward propagation
        '''
        self.__cache['A0'] = X

        for lay in range(self.__L):
            W = self.__weights[f'W{lay+1}']
            b = self.__weights[f'b{lay+1}']
            Z = np.matmul(W, self.__cache[f'A{lay}']) + b
            A = 1 / (1 + self.sigmoid(-Z))
            self.__cache[f'A{lay+1}'] = A
        return self.cache[f'A{self.__L}'], self.__cache

