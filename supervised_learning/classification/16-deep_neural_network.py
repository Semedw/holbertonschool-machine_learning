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
        
        self.L = len(layers) # the number of layers in the neural network
        self.cache = {} # a dict to hold all intermediary values of the network
        self.weights = {}
        for l in range(self.L):
            
            if not isinstance(layers[l], int) or layers[l] < 1:
                raise TypeError('layers must be a list of positive integers')
            
            # Determine the input size for the current layer (n_{l-1})
            # For the first layer, it's nx. For others, it's the previous layer's size.
            n_prev = nx if l == 0 else layers[l - 1]
            
            # Key for the dictionary (1-indexed based on your prompt)
            layer_num = l + 1
            
            # He et al. Initialization: W = randn * sqrt(2 / n_prev)
            self.weights[f'W{layer_num}'] = np.random.randn(layers[l], n_prev) * np.sqrt(2 / n_prev)
            
            # Bias Initialization: Zeros
            self.weights[f'b{layer_num}'] = np.zeros((layers[l], 1))
