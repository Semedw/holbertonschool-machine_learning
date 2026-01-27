#!/usr/bin/env python3
"""
finding determinant
"""


def determinant(matrix):
    """
    finding the determinant of matrix
    """
    if isinstance(matrix, list):
        for i in matrix:
            if not isinstance(i, list):
                raise TypeError('matrix must be a list of lists')
        k = len(matrix)
        for i in matrix:
            if len(i) == 0:
                return 1
            if len(i) != k:
                raise ValueError('matrix must be a square matrix')
        if len(matrix) == 1:
            return matrix[0][0]
        if len(matrix) == 2:
            det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
            return det
    else:
        raise TypeError('matrix must be a list of lists')