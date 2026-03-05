#!/usr/bin/env python3
'''
neural network
'''


import numpy as np
import matplotlib.pyplot as plt


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
        A = self.forward_prop(X)[1]
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        '''
        calculating one pass of gradient descent
        '''

        error2 = A2 - Y
        error1 = np.dot(self.W2.T, error2) * (A1 * (1 - A1))

        m = Y.shape[1]
        # __W1, __b1
        gradient_weight1 = (1 / m) * np.matmul(error1, X.T)
        gradient_bias1 = (1 / m) * np.sum(error1, axis=1, keepdims=True)
        self.__W1 = self.__W1 - alpha * gradient_weight1
        self.__b1 = self.__b1 - alpha * gradient_bias1
        # __W2, __b2
        gradient_weight2 = (1 / m) * np.matmul(error2, A1.T)
        gradient_bias2 = (1 / m) * np.sum(error2, axis=1, keepdims=True)
        self.__W2 = self.__W2 - alpha * gradient_weight2
        self.__b2 = self.__b2 - alpha * gradient_bias2

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        '''
        training the neural network
        '''
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        x = []
        y = []

        if verbose == True or graph == True:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step <= 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        
        for iteration in range(iterations+1):
            A1, A2 = self.forward_prop(X)
            cost = self.cost(Y, A2)
            if verbose == True:
                if iteration % step == 0:
                    print(f'Cost after {iteration} iterations: {cost}')
                    if graph == True:
                        x.append(iteration)
                        y.append(cost)
            self.gradient_descent(X, Y, A1, A2, alpha)

        if graph == True:
            plt.figure(figsize=(6.4, 4.8))
            plt.plot(x, y)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training cost')
            plt.show()
        
        return self.evaluate(X, Y)
