#!/usr/bin/env python3
'''
neural network
'''


import numpy as np


class NeuralNetwork:
    '''
    neural network class
    '''
    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        '''
        w1 getter
        '''
        return self.__W1

    @property
    def W2(self):
        '''
        w1 getter
        '''
        return self.__W2

    @property
    def b1(self):
        '''
        w1 getter
        '''
        return self.__b1

    @property
    def b2(self):
        '''
        w1 getter
        '''
        return self.__b2

    @property
    def A1(self):
        '''
        w1 getter
        '''
        return self.__A1

    @property
    def A2(self):
        '''
        w1 getter
        '''
        return self.__A2

    @staticmethod
    def sigmoid(x):
        '''
        sigmoid function
        '''
        return 1 / (1 + np.exp(-x))

    def forward_prop(self, X):
        '''
        calculating forward propagation(output)
        '''
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(Z1)
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(Z2)
        return self.__A1, self.__A2

    def cost(self, Y, A):
        '''
        calculating neural network cost
        '''
        m = Y.shape[1]
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001-A))
        cost = (1 / m) * np.sum(loss)
        return cost

    def evaluate(self, X, Y):
        '''
        evaluetes the neural network's predictions
        '''
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost
