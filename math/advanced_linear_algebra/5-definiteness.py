#!/usr/bin/env
"""
finding definiteness of matrix
"""

import numpy as np

def transpose(matrix):
    """
    Docstring for transpose

    :param matrix: matrix
    """
    new_mat = [[0 for i in range(len(matrix[0]))] for i in range(len(matrix))]
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            new_mat[col][row] = matrix[row][col]
    return new_mat


def definiteness(matrix):
    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')
    k = len(matrix[0])
    if transpose(matrix) != matrix:
        return None
    for i in matrix:
        if len(i) != k:
            return None

    eigenvalues = np.linalg.eigvals(matrix)
    signs = []
    for i in eigenvalues:
        if i > 0:
            signs.append(1)
        elif i < 0:
            signs.append(-1)
        else:
            signs.append(0)
    if signs.count(1) > 0 and signs.count(-1) > 0:
        return 'Indefinite'
    elif signs.count(1) > 0 and signs.count(0) > 0:
        return 'Positive semi-definite'
    elif signs.count(-1) > 0 and signs.count(0) > 0:
        return 'Negative semi-definite'
    elif signs.count(1) > 0:
        return 'Positive definite'
    elif signs.count(-1) > 0:
        return 'Negative definite'
    else:
        return None