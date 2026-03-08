#!/usr/bin/env python3
"""
Deep Neural Network module
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing multiclass classification.
    """

    def __init__(self, nx, layers):
        '''
        nx - the number of input features -> int
        layers - the number of nodes in eahc layer of the network -> List
        e.g. layers[0] represents the the number of nodes in the first layer
        '''
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

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
    def L(self):
        """
        Retrieves the number of layers in the neural network.

        Returns:
            int: Total number of layers.
        """
        return self.__L

    @property
    def cache(self):
        """
        Retrieves the cache dictionary.

        Returns:
            dict: Dictionary containing intermediary activations.
        """
        return self.__cache

    @property
    def weights(self):
        """
        Retrieves the weights dictionary.

        Returns:
            dict: Dictionary containing weights and biases.
        """
        return self.__weights

    @staticmethod
    def sigmoid(x):
        """
        Computes the sigmoid activation function.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Sigmoid activated output.
        """
        return 1 / (1 + np.exp(-x))

    def forward_prop(self, X):
        """
        Performs forward propagation through the network.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).

        Returns:
            tuple:
                numpy.ndarray: Output of the neural network.
                dict: Updated cache containing all activations.
        """
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
        """
        Computes the categorical cross-entropy cost.

        Args:
            Y (numpy.ndarray): Correct labels in one-hot format.
            A (numpy.ndarray): Activated output from the network.

        Returns:
            float: Computed cost.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the predictions of the neural network.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): Correct labels.

        Returns:
            tuple:
                numpy.ndarray: Predicted labels in one-hot format.
                float: Cost of the network.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        prediction = np.eye(A.shape[0])[np.argmax(A, axis=0)].T

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Performs one pass of gradient descent.

        Args:
            Y (numpy.ndarray): Correct labels.
            cache (dict): Dictionary containing forward propagation values.
            alpha (float): Learning rate.
        """
        m = Y.shape[1]
        dZ = cache["A{}".format(self.__L)] - Y

        for lay in range(self.__L, 0, -1):
            A_prev = cache["A{}".format(lay - 1)]
            W = self.__weights["W{}".format(lay)]

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if lay > 1:
                A_curr = cache["A{}".format(lay - 1)]
                dZ = np.matmul(W.T, dZ) * A_curr * (1 - A_curr)

            self.__weights["W{}".format(lay)] -= alpha * dW
            self.__weights["b{}".format(lay)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): Correct labels.
            iterations (int): Number of training iterations.
            alpha (float): Learning rate.
            verbose (bool): If True prints cost during training.
            graph (bool): If True plots the training cost.
            step (int): Step interval for printing and plotting.

        Returns:
            tuple:
                numpy.ndarray: Predicted labels.
                float: Final cost.
        """
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
            A, _ = self.forward_prop(X)
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
        """
        Saves the neural network object to a file using pickle.

        Args:
            filename (str): Name of the file to save the object.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a DeepNeuralNetwork object from a pickle file.

        Args:
            filename (str): File containing the saved object.

        Returns:
            DeepNeuralNetwork or None: Loaded object or None if file
            does not exist.
        """
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
