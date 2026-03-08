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

    def __init__(self, nx, layers):
        '''
        nx - the number of input features -> int
        layers - the number of nodes in each layer of the network -> List
        e.g. layers[0] represents the number of nodes in the first layer
        '''
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)  # the number of layers in the neural network
        self.__cache = {}  # to hold all intermediary values of the network
        self.__weights = {}

        for lay in range(self.__L):
            if not isinstance(layers[lay], int) or layers[lay] < 1:
                raise TypeError("layers must be a list of positive integers")

            if lay == 0:
                He_val = np.random.randn(layers[lay], nx) * np.sqrt(2 / nx)
                self.__weights["W" + str(lay + 1)] = He_val
            else:
                He_val1 = np.random.randn(layers[lay], layers[lay - 1])
                He_val2 = np.sqrt(2 / layers[lay - 1])
                He_val3 = He_val1 * He_val2
                self.__weights["W" + str(lay + 1)] = He_val3

            self.__weights["b" + str(lay + 1)] = np.zeros((layers[lay], 1))

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
        self.__cache["A0"] = X

        for lay in range(self.__L):
            W = self.__weights["W{}".format(lay + 1)]
            b = self.__weights["b{}".format(lay + 1)]

            Z = np.matmul(W, self.__cache["A{}".format(lay)]) + b

            if lay == self.__L - 1:
                e = np.exp(Z)
                A = e / np.sum(e, axis=0, keepdims=True)
            else:
                A = self.sigmoid(Z)

            self.__cache["A{}".format(lay + 1)] = A

        return A, self.__cache

    def cost(self, Y, A):
        '''
        calculates the cost of model using logistic regression
        Y - correct labels
        A - activated outputs
        '''
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):
        '''
        evaluates the neural network's predictions
        X - input data
        Y - correct labels
        '''
        A = self.forward_prop(X)[0]
        cost = self.cost(Y, A)

        prediction = np.eye(A.shape[0])[np.argmax(A, axis=0)].T
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        '''
        calculating one pass of gradient descent on the neural network
        Y - correct labels
        cache - a dict containing all the intermediary values of the network
        alpha - learning rate
        '''
        m = Y.shape[1]

        # Activation of the final layer, aka the output
        dzh = cache["A{}".format(self.__L)] - Y

        for layer in range(self.__L, 0, -1):

            # Activation of the previous layer
            A_prev = cache["A{}".format(layer - 1)]

            # Weights of the current layer
            W = self.__weights["W{}".format(layer)]

            dwh = np.matmul(dzh, A_prev.T) / m
            dbh = np.sum(dzh, axis=1, keepdims=True) / m

            if layer > 1:
                A_curr = cache["A{}".format(layer - 1)]
                dzl = np.matmul(W.T, dzh) * A_curr * (1 - A_curr)

            # Updating weights and biases
            self.__weights["W{}".format(layer)] -= alpha * dwh
            self.__weights["b{}".format(layer)] -= alpha * dbh

            if layer > 1:
                dzh = dzl

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        '''
        trains the deep neural network
        X - input data
        Y - correct labels
        iterations - the number of iterations to train over
        alpha - the training rate
        verbose - defines whether or not to print the info about the training
        graph - defines whether or not to graph the info about the training
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
                    print(f"Cost after {i} iterations: {cost}")

            if i < iterations:
                self.gradient_descent(Y, self.__cache, alpha)

        if graph:
            plt.figure(figsize=(6.4, 4.8))
            plt.plot(x, y)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        '''
        saves the instance object to a file in pickle format
        '''
        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, mode="wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        '''
        loading the deep neural network object
        '''
        try:
            with open(filename, mode="rb") as file:
                k = pickle.load(file)
                return k
        except FileNotFoundError:
            return None
