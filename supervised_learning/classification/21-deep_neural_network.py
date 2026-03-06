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
            A = self.sigmoid(Z)
            self.__cache[f'A{lay+1}'] = A
        return self.__cache[f'A{self.__L}'], self.__cache

    def cost(self, Y, A):
        '''
        calculates the cost of model using logistic regression
        Y - correct labels
        A - activated outputs
        '''

        m = Y.shape[1]
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001-A))
        cost = (1 / m) * np.sum(loss)
        return cost

    def evaluate(self, X, Y):
        '''
        evaluates the neural network's predictions
        X - input data
        Y - correct labels
        '''

        A = self.forward_prop(X)[0]
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        '''
        calculating one pass of gradient descent on the neural network
        Y - correct labels
        cache - a dict containing all the intermediary values of the network
        alpha - learning rate
        '''
        m = Y.shape[1]
        # Start the error (dz) with the output layer
        # Assumes Softmax/Cross-Entropy or Sigmoid/MSE derivative
        dz = cache[f'A{self.__L}'] - Y 

        for lay in reversed(range(1, self.__L + 1)):
            # 1. Calculate Gradients
            # A{lay-1} is the input to the current layer
            dW = np.matmul(dz, cache[f'A{lay-1}'].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            # 2. Backpropagate the error to the previous layer (if not at the input)
            if lay > 1:
                # Derivative of Sigmoid: a * (1 - a)
                # We use the activation of the PREVIOUS layer to prep for the next loop
                a_prev = cache[f'A{lay-1}']
                da_prev = a_prev * (1 - a_prev)
                dz = np.matmul(self.weights[f'W{lay}'].T, dz) * da_prev

            # 3. Update Weights
            self.weights[f'W{lay}'] -= alpha * dW
            self.weights[f'b{lay}'] -= alpha * db
