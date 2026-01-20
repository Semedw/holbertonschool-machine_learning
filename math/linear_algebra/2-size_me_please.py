#!/usr/bin/env python3
"""
sizing the matrix
"""

import numpy as np


def matrix_shape(matrix):
    """
    matrix shape
    """
    matrix = np.array(matrix)
    shape = matrix.shape
    return list(shape)
