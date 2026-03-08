#!/usr/bin/env python3
"""
Deep Neural Network
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """builds a deep neural network"""

    def __init__(self, nx, layers):

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

            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

            layer = i + 1

            if i == 0:
                w = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                w = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2 / layers[i-1])

            self.__weights[f"W{layer}"] = w
            self.__weights[f"b{layer}"] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def forward_prop(self, X):

        self.__cache["A0"] = X

        for l in range(1, self.__L + 1):

            W = self.__weights[f"W{l}"]
            b = self.__weights[f"b{l}"]

            Z = np.matmul(W, self.__cache[f"A{l-1}"]) + b

            if l == self.__L:
                # softmax with numerical stability
                Z = Z - np.max(Z, axis=0, keepdims=True)
                exp = np.exp(Z)
                A = exp / np.sum(exp, axis=0, keepdims=True)
            else:
                A = self.sigmoid(Z)

            self.__cache[f"A{l}"] = A

        return A, self.__cache

    def cost(self, Y, A):

        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):

        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        prediction = np.eye(A.shape[0])[np.argmax(A, axis=0)].T

        return prediction, cost

    def gradient_descent(self, Y, alpha=0.05):

        m = Y.shape[1]

        dZ = self.__cache[f"A{self.__L}"] - Y

        for l in reversed(range(1, self.__L + 1)):

            A_prev = self.__cache[f"A{l-1}"]
            W = self.__weights[f"W{l}"]

            dW = (dZ @ A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if l > 1:
                A_curr = self.__cache[f"A{l-1}"]
                dZ = (W.T @ dZ) * A_curr * (1 - A_curr)

            self.__weights[f"W{l}"] -= alpha * dW
            self.__weights[f"b{l}"] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        costs = []
        steps = []

        for i in range(iterations + 1):

            A, _ = self.forward_prop(X)
            cost = self.cost(Y, A)

            if i % step == 0 or i == iterations:
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

                costs.append(cost)
                steps.append(i)

            if i < iterations:
                self.gradient_descent(Y, alpha)

        if graph:
            plt.plot(steps, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):

        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):

        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
