#!/usr/bin/env python3
'''
deep neural network
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    '''
    building a deep neural network
    '''

    def __init__(self, nx, layers, activation='sig'):
        '''
        nx - number of input features -> int
        layers - list containing number of nodes in each layer
        activation - activation function for hidden layers
        '''

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        if activation != 'sig' and activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__activation = activation
        self.__L = len(layers)  # number of layers
        self.__cache = {}  # storing intermediary values
        self.__weights = {}  # storing weights and biases

        for i in range(self.__L):

            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                w = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                w = np.random.randn(layers[i], layers[i - 1]) * \
                    np.sqrt(2 / layers[i - 1])

            self.__weights["W{}".format(i + 1)] = w
            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))

    @property
    def activation(self):
        '''
        activation getter
        '''
        return self.__activation

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
        sigmoid activation function
        '''
        return 1 / (1 + np.exp(-x))

    def forward_prop(self, X):
        '''
        forward propagation
        '''

        self.__cache["A0"] = X

        for lay in range(self.__L):

            W = self.__weights["W{}".format(lay + 1)]
            b = self.__weights["b{}".format(lay + 1)]

            Z = np.matmul(W, self.__cache["A{}".format(lay)]) + b

            if lay == self.__L - 1:
                e = np.exp(Z)
                A = e / np.sum(e, axis=0, keepdims=True)
            else:
                if self.activation == 'sig':
                    A = self.sigmoid(Z)
                else:
                    A = np.tanh(Z)

            self.__cache["A{}".format(lay + 1)] = A

        return A, self.__cache

    def cost(self, Y, A):
        '''
        calculating cost using categorical cross entropy
        Y - correct labels
        A - activated output
        '''

        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):
        '''
        evaluating neural network predictions
        X - input data
        Y - correct labels
        '''

        A = self.forward_prop(X)[0]
        cost = self.cost(Y, A)

        prediction = np.eye(A.shape[0])[np.argmax(A, axis=0)].T

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        '''
        one pass of gradient descent
        Y - correct labels
        cache - intermediary values
        alpha - learning rate
        '''

        m = Y.shape[1]
        dZ = cache["A{}".format(self.__L)] - Y

        for lay in range(self.__L, 0, -1):

            A_prev = cache["A{}".format(lay - 1)]
            W = self.__weights["W{}".format(lay)]

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if lay > 1:

                A_curr = cache["A{}".format(lay - 1)]

                if self.activation == 'sig':
                    dZ = np.matmul(W.T, dZ) * A_curr * (1 - A_curr)
                else:
                    dZ = np.matmul(W.T, dZ) * (1 - A_curr ** 2)

            self.__weights["W{}".format(lay)] -= alpha * dW
            self.__weights["b{}".format(lay)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        '''
        training the deep neural network
        X - input data
        Y - correct labels
        iterations - number of iterations
        alpha - learning rate
        '''

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")

        if iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")

        if alpha < 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:

            if not isinstance(step, int):
                raise TypeError("step must be an integer")

            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        x = []
        y = []

        for i in range(iterations + 1):

            A = self.forward_prop(X)[0]
            cost = self.cost(Y, A)

            if i % step == 0:

                x.append(i)
                y.append(cost)

                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))

            if i < iterations:
                self.gradient_descent(Y, self.__cache, alpha)

        if graph:

            plt.plot(x, y)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        '''
        saving neural network object using pickle
        '''

        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        '''
        loading neural network object from pickle file
        '''

        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
