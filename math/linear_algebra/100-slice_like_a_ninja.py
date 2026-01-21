#!/usr/bin/env python3
"""
slice like a ninja
"""
# import numpy as np


def np_slice(matrix, axes={}):
    """
    slicing matrix along specific axes
    """
    newMat = matrix
    result = []
    for key, value in axes.items(): 
        # sec = slice(*value)
        newMat = newMat.take(indices=range(*value), axis=key)
    return newMat

# mat1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
# print(np_slice(mat1, axes={1: (1, 3)}))
