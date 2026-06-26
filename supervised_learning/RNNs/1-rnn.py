#!/usr/bin/env python3
"""A module to Create the class RNNCell"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """performs forward propagation for a simple RNN"""
    caches = []
    t, m, i = X.shape
    time_s = range(t)

    _, h = h_0.shape

    H = np.zeros((t+1, m, h))
    H[0, :, :] = h_0

    for ts in time_s:
        h_next, y_pred = rnn_cell.forward(H[ts], X[ts])
        H[ts+1, :, :] = h_next
        caches.append(y_pred)
    caches = np.array(caches)
    return H, caches
