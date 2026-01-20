#!/usr/bin/env python3
"""
i'll use my scale
"""


def np_shape(matrix):
    """
    calculating the shape of matrix
    """
    shape = []
    m = matrix
    while isinstance(m, list):
        shape.append(len(m))
        if len(m) != 0:
            m = m[0]
        else:
            break
    return tuple(shape)
