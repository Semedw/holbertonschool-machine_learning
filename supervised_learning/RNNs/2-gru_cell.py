#!/usr/bin/env python3
"""GRU cell for a RRN"""
import numpy as np


class GRUCell:
    """ Gated recurrent unit """

    def __init__(self, i, h, o):
        """constructor"""
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bz = np.zeros((1, h))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """softmax function"""
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def sigmoid(self, x):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, x_t):
        """forward propagation"""
        h_x = np.concatenate((h_prev.T, x_t.T), axis=0)
        zt = self.sigmoid((h_x.T @ self.Wz) + self.bz)
        rt = self.sigmoid((h_x.T @ self.Wr) + self.br)
        h_x = np.concatenate(((rt * h_prev).T, x_t.T), axis=0)
        ht_c = np.tanh((h_x.T @ self.Wh) + self.bh)
        h_next = (1 - zt) * h_prev + zt * ht_c
        y = self.softmax((h_next @ self.Wy) + self.by)

        return h_next, y
