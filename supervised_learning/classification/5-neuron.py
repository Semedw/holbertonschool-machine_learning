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
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        '''
        loss (cost) function
        '''
        m = Y.shape[1]
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001-A))
        cost = (1 / m) * np.sum(loss)
        return cost

    def evaluate(self, X, Y):
        '''
        Evaluates the neuron's predictions
        '''
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        '''
        this function calculates one pass of gradient descent
        '''
        h = np.dot(self.__W.T, X)

        error = h - Y

        m = len(Y)
        gradient = np.dot(X.T, error) / m
        self.__W = self.__W - alpha * gradient
        self.__b = self.__b - alpha * gradient
