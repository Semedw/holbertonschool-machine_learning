#!/usr/bin/env python3
"""
sizing the matrix
"""


def matrix_shape(matrix):
    """
    matrix shape
    """
    m = matrix
    shape = [len(m)]
    while isinstance(m[0], list):
        shape.append(len(m[0]))
        m = m[0]
    return shape
